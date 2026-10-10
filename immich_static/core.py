"""
immich_static.core — Shared engine for Immich static video detection, frame extraction, and checkpointing.
"""

import atexit
from contextlib import contextmanager
import json
import os

# Suppress low-level FFmpeg / OpenCV C-library decoder warning spam on stderr
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("AV_LOG_FORCE_NOCOLOR", "1")
try:
    import cv2
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

import shutil
import signal
import sqlite3
import subprocess
import sys
import termios
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ─────────────────────────────────────────────
# AUTO-LOAD .ENV FILE IF PRESENT
# ─────────────────────────────────────────────

def get_project_root() -> Path:
    """Returns the resolved project root directory."""
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd(),
        pkg_dir.parent,
    ]
    for c in candidates:
        if (c / ".git").is_dir() or ((c / "pyproject.toml").is_file() and (c / "immich_static").is_dir()):
            return c.resolve()
    return Path.cwd().resolve()


def load_dotenv(dotenv_path: Optional[Path] = None):
    """
    Loads key-value pairs from .env files into os.environ if not already set.
    Searches current directory, package project root, user config, and parent directories.
    Resolves relative path variables (DB_PATH, EXTRACT_OUTPUT_DIR) to absolute paths.
    """
    pkg_root = get_project_root()

    candidates: List[Path] = []
    if dotenv_path:
        candidates.append(Path(dotenv_path).resolve())

    # 1. Current working directory .env
    candidates.append(Path.cwd() / ".env")
    # 2. Package / project root .env (where immich-static repo is located)
    candidates.append(pkg_root / ".env")
    # 3. User configuration directory ~/.config/immich-static/.env
    candidates.append(Path.home() / ".config" / "immich-static" / ".env")
    # 4. User home directory fallback
    candidates.append(Path.home() / ".immich-static.env")

    # 5. Search upwards from cwd to home directory
    curr = Path.cwd().resolve()
    for parent in curr.parents:
        if parent == Path.home() or parent == Path("/"):
            break
        cand = parent / ".env"
        if cand not in candidates:
            candidates.append(cand)

    for env_file in candidates:
        if env_file.is_file():
            try:
                base_dir = env_file.resolve().parent
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if not k:
                            continue

                        # Resolve relative paths in .env to absolute paths based on .env location
                        if k in ("DB_PATH", "EXTRACT_OUTPUT_DIR") and v and not os.path.isabs(v):
                            v = str((base_dir / v).resolve())

                        if k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass

    # Bidirectional alias support: IMMICH_URL <-> IMMICH_API_URL
    if "IMMICH_API_URL" not in os.environ and "IMMICH_URL" in os.environ:
        os.environ["IMMICH_API_URL"] = os.environ["IMMICH_URL"]
    if "IMMICH_URL" not in os.environ and "IMMICH_API_URL" in os.environ:
        os.environ["IMMICH_URL"] = os.environ["IMMICH_API_URL"]

    # Provide absolute fallback paths for DB and extracted frames if not defined
    if "DB_PATH" not in os.environ:
        os.environ["DB_PATH"] = str((pkg_root / CHECKPOINT_FILENAME).resolve())
    if "EXTRACT_OUTPUT_DIR" not in os.environ:
        os.environ["EXTRACT_OUTPUT_DIR"] = str((pkg_root / "extracted_frames").resolve())


load_dotenv()

# ─────────────────────────────────────────────
# ENVIRONMENT & HARDWARE DETECTION
# ─────────────────────────────────────────────

IS_TERMUX = "com.termux" in os.environ.get("PREFIX", "") or os.path.exists("/data/data/com.termux")
CPU_COUNT = os.cpu_count() or 4
MAX_WORKERS = min(CPU_COUNT * 2, 16) if not IS_TERMUX else min(CPU_COUNT, 2)

ENV = {
    "is_termux": IS_TERMUX,
    "cpu_count": CPU_COUNT,
    "max_workers": MAX_WORKERS,
}

VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".mov", ".mkv", ".avi", ".webm",
    ".m4v", ".3gp", ".wmv", ".flv", ".ts",
    ".mts", ".m2ts"
})

CHECKPOINT_FILENAME = "detection_checkpoint.sqlite"

# Analysis parameters matching StaticSorter
ANALYSIS_WIDTH = 320
GRID_ROWS = 6
GRID_COLS = 6
ZONE_MOTION_THRESH = 5.0

LOG_FIELDS = [
    "filename",
    "original_file_name",
    "asset_id",
    "decision",
    "final_confidence",
    "global_motion_score",
    "active_zone_ratio",
    "max_zone_motion",
    "heuristic_score",
    "duration_s",
    "width",
    "height",
    "fps",
    "frames_sampled",
    "error",
]


class SensitivityConfig:
    """Thresholds for detection sensitivity levels matching StaticSorter."""
    def __init__(
        self,
        global_motion_static: float,
        global_motion_review: float,
        active_zone_ratio: float,
        confidence_static: float,
        confidence_review: float,
    ):
        self.global_motion_static = float(global_motion_static)
        self.global_motion_review = float(global_motion_review)
        self.active_zone_ratio = float(active_zone_ratio)
        self.confidence_static = float(confidence_static)
        self.confidence_review = float(confidence_review)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def __repr__(self) -> str:
        return (
            f"SensitivityConfig(static_motion<={self.global_motion_static}, "
            f"review_motion<={self.global_motion_review}, active_zones<={self.active_zone_ratio}, "
            f"conf_static>={self.confidence_static}, conf_review>={self.confidence_review})"
        )


SENSITIVITY_PRESETS: Dict[str, SensitivityConfig] = {
    "low": SensitivityConfig(
        global_motion_static=3.0,
        global_motion_review=8.0,
        active_zone_ratio=0.30,
        confidence_static=0.75,
        confidence_review=0.45,
    ),
    "medium": SensitivityConfig(
        global_motion_static=4.5,
        global_motion_review=12.0,
        active_zone_ratio=0.25,
        confidence_static=0.65,
        confidence_review=0.40,
    ),
    "high": SensitivityConfig(
        global_motion_static=7.0,
        global_motion_review=18.0,
        active_zone_ratio=0.35,
        confidence_static=0.55,
        confidence_review=0.35,
    ),
}


def format_bytes(bytes_count: Any) -> str:
    """Converts raw byte count into human readable string (KB, MB, GB, TB)."""
    if not bytes_count:
        return "0 B"
    try:
        val = float(bytes_count)
    except (ValueError, TypeError):
        return "0 B"
    if val <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while val >= 1024.0 and i < len(units) - 1:
        val /= 1024.0
        i += 1
    return f"{val:.2f} {units[i]}"


def format_duration(seconds: float) -> str:
    """Converts seconds into formatted string (e.g. 2h 15m 30s)."""
    seconds = max(0.0, float(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h)}h {int(m)}m {int(s)}s"
    if m > 0:
        return f"{int(m)}m {int(s)}s"
    return f"{int(s)}s"


# ─────────────────────────────────────────────
# PROGRESS BAR FALLBACK
# ─────────────────────────────────────────────

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class FallbackBar:
    """Simple ASCII fallback progress bar if tqdm is unavailable."""
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.desc = desc
        self.count = 0
        self.start_time = time.time()
        self._print()

    def update(self, n: int = 1):
        self.count += n
        self._print()

    def set_postfix(self, d: Dict[str, Any]):
        pass

    def _print(self):
        pct = (self.count / self.total * 100) if self.total > 0 else 0
        bar_len = 30
        filled = int(bar_len * (self.count / self.total)) if self.total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stderr.write(f"\r{self.desc}: [{bar}] {self.count}/{self.total} ({pct:.1f}%)")
        sys.stderr.flush()
        if self.count >= self.total:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def close(self):
        if self.count < self.total:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def make_bar(total: int, desc: str = "Processing", unit: str = "video"):
    """Returns a tqdm or Fallback progress bar."""
    if HAS_TQDM and sys.stderr.isatty():
        return tqdm(total=total, desc=desc, unit=unit)
    return FallbackBar(total=total, desc=desc)


# ─────────────────────────────────────────────
# TERMINAL STATE & SIGNAL HANDLING
# ─────────────────────────────────────────────

_original_termios = None
_interrupt_event = threading.Event()
_hard_stop_event = threading.Event()


def save_terminal_state():
    global _original_termios
    if sys.stdin.isatty():
        try:
            _original_termios = termios.tcgetattr(sys.stdin)
        except Exception:
            _original_termios = None


def restore_terminal_state():
    if _original_termios is not None and sys.stdin.isatty():
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _original_termios)
        except Exception:
            pass


def handle_sigint(sig, frame):
    if _interrupt_event.is_set():
        print("\n\n⛔ Force stopping immediately...")
        _hard_stop_event.set()
        restore_terminal_state()
        sys.exit(130)
    print("\n\n⚠️  Gracefully finishing active workers and saving checkpoint (press Ctrl+C again to force stop)...")
    _interrupt_event.set()


def setup_interrupt_handlers():
    save_terminal_state()
    atexit.register(restore_terminal_state)
    signal.signal(signal.SIGINT, handle_sigint)


# ─────────────────────────────────────────────
# DEPENDENCY CHECK
# ─────────────────────────────────────────────

def check_system_dependencies(need_cv: bool = True):
    """Verifies that ffmpeg, ffprobe, and required python packages are available."""
    missing = []
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            missing.append(f"System tool: {tool}")
    if need_cv:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as e:
            missing.append(f"Python package: {e}")

    if not missing:
        return

    print("❌ Missing required dependencies:")
    for m in missing:
        print(f"   • {m}")
    print("\nPlease install the missing dependencies:")
    print("   ffmpeg  → apt install ffmpeg / brew install ffmpeg / pkg install ffmpeg")
    print("   python  → pip install opencv-python-headless numpy tqdm")
    sys.exit(1)


# ─────────────────────────────────────────────
# VIDEO PROBE & SAMPLING
# ─────────────────────────────────────────────

def get_video_metadata(video_path: Path) -> Dict[str, Any]:
    """Uses ffprobe to extract video stream information."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames:format=duration",
        "-of", "json",
        str(video_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {res.stderr.strip()}")

    data = json.loads(res.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    width = 0
    height = 0
    fps = 0.0
    duration = 0.0

    if streams:
        v_stream = streams[0]
        width = int(v_stream.get("width") or 0)
        height = int(v_stream.get("height") or 0)
        r_rate = v_stream.get("r_frame_rate", "0/1")
        if "/" in r_rate:
            num, den = r_rate.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        else:
            fps = float(r_rate) if r_rate else 0.0
        duration = float(v_stream.get("duration") or 0.0)

    if duration == 0.0:
        duration = float(fmt.get("duration") or 0.0)

    return {
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "duration": round(duration, 2),
    }


def sample_count_for_duration(duration: float) -> int:
    """Determine number of frames to sample during classification matching StaticSorter."""
    if duration < 3:
        return 2
    if duration < 5:
        return 3
    if duration < 10:
        return 4
    if duration < 30:
        return 5
    if duration < 60:
        return 8
    return 16


def extract_sparse_frames(
    video_path: Path,
    duration: float,
    count: Optional[int] = None,
    target_width: int = ANALYSIS_WIDTH,
) -> List[Any]:
    """
    Extracts sparse downscaled frames for rapid motion analysis.
    Uses exact frame index sampling when available, timestamp sampling as fallback,
    and sequential extraction if seeks fail.
    """
    import cv2

    video_path = Path(video_path)
    if duration <= 0:
        duration = 1.0

    num_samples = count if count is not None else sample_count_for_duration(duration)

    frames = []
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

            # 1. Primary: If total_frames is known and valid, sample by exact frame index
            if total_frames >= 2:
                indices = [
                    max(0, min(total_frames - 1, int(total_frames * (i + 0.5) / num_samples)))
                    for i in range(num_samples)
                ]
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        if w > 0 and h > 0:
                            scale = target_width / w
                            new_h = max(1, int(h * scale))
                            frame = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)
                        frames.append(frame)

            # 2. Secondary: If index seeking yielded insufficient frames, try timestamp seeking
            if len(frames) < 2:
                frames = []
                margin = max(0.04, min(0.5, duration * 0.05))
                t_start = margin
                t_end = max(t_start + 0.04, duration - margin)
                step = (t_end - t_start) / max(1, num_samples - 1)
                timestamps = [t_start + i * step for i in range(num_samples)]

                for ts in timestamps:
                    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        frame_idx = int(ts * fps)
                        if total_frames > 0:
                            frame_idx = min(total_frames - 1, frame_idx)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()

                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        if w > 0 and h > 0:
                            scale = target_width / w
                            new_h = max(1, int(h * scale))
                            frame = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)
                        frames.append(frame)

            # 3. Tertiary: Sequential read from beginning (for variable-framerate or stream-based videos)
            if len(frames) < 2:
                frames = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                while len(frames) < max(2, num_samples):
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    h, w = frame.shape[:2]
                    if w > 0 and h > 0:
                        scale = target_width / w
                        new_h = max(1, int(h * scale))
                        frame = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)
                    frames.append(frame)
        finally:
            cap.release()

    if len(frames) < 2:
        timestamps = [duration * (i + 0.5) / num_samples for i in range(num_samples)]
        frames = _ffmpeg_extract_sparse(video_path, timestamps, target_width)

    return frames


def _ffmpeg_extract_sparse(video_path: Path, timestamps: List[float], target_width: int = ANALYSIS_WIDTH) -> List[Any]:
    """Fallback sparse extractor using FFmpeg CLI pipe."""
    import cv2
    import numpy as np

    frames = []
    for ts in timestamps:
        cmd = [
            "ffmpeg",
            "-ss", f"{ts:.3f}",
            "-i", str(video_path),
            "-vframes", "1",
            "-vf", f"scale={target_width}:-2",
            "-f", "image2pipe",
            "-vcodec", "png",
            "-"
        ]
        res = subprocess.run(cmd, capture_output=True, check=False)
        if res.returncode == 0 and res.stdout:
            nparr = np.frombuffer(res.stdout, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
    return frames


# ─────────────────────────────────────────────
# 3-LAYER DETECTION & GRID ANALYSIS (STATICSORTER ENGINE)
# ─────────────────────────────────────────────

def compute_global_motion(frames: List[Any]) -> float:
    """
    Layer 1: Compute average pixel intensity difference across consecutive frames (0-255 scale).
    """
    import cv2
    import numpy as np

    if len(frames) < 2:
        return 0.0

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f for f in frames]
    diffs = []
    for i in range(len(grays) - 1):
        diff = cv2.absdiff(grays[i], grays[i + 1])
        diffs.append(float(np.mean(diff)))

    return float(np.mean(diffs)) if diffs else 0.0


def compute_spatial_zone_ratio(
    frames: List[Any],
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
    thresh: float = ZONE_MOTION_THRESH,
) -> Tuple[float, float]:
    """
    Layer 2: Divide frame into 6x6 grid (36 zones) and compute fraction of active motion zones.
    Returns: (active_zone_ratio, max_zone_motion)
    """
    import cv2
    import numpy as np

    if len(frames) < 2:
        return 0.0, 0.0

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f for f in frames]
    h, w = grays[0].shape[:2]
    cell_h = h // rows
    cell_w = w // cols

    if cell_h == 0 or cell_w == 0:
        return 0.0, 0.0

    active_zones = 0
    total_zones = rows * cols
    max_zone_motion = 0.0

    diff_maps = [cv2.absdiff(grays[i], grays[i + 1]) for i in range(len(grays) - 1)]

    for r in range(rows):
        for c in range(cols):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h if r < rows - 1 else h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w if c < cols - 1 else w

            cell_diffs = [float(np.mean(dm[y1:y2, x1:x2])) for dm in diff_maps]

            if cell_diffs:
                avg_zone_motion = float(np.mean(cell_diffs))
                if avg_zone_motion > max_zone_motion:
                    max_zone_motion = avg_zone_motion
                if avg_zone_motion > thresh:
                    active_zones += 1

    active_ratio = active_zones / total_zones if total_zones > 0 else 0.0
    return round(active_ratio, 3), round(max_zone_motion, 4)


def compute_heuristics(meta: Any = None) -> float:
    """Deprecated: Metadata heuristic guesswork has been removed in favor of direct visual motion analysis."""
    return 0.5


def compute_confidence(
    global_motion: float,
    zone_ratio: float,
    heuristic: Optional[float] = None,
    cfg: Optional[SensitivityConfig] = None,
    duration: float = 0.0,
    **kwargs: Any,
) -> float:
    """
    Combine scores into a single confidence rating (0.0 to 1.0) based purely on visual ground truth:
    60% Global Motion + 40% Spatial Active Zones (36-zone 6x6 grid analysis).
    """
    import numpy as np

    if cfg is None:
        cfg = SENSITIVITY_PRESETS["medium"]

    # 1. Motion score: <= static_thresh -> 1.0, >= review_thresh -> 0.0
    static_thresh = cfg.global_motion_static
    review_thresh = max(static_thresh + 0.1, cfg.global_motion_review)

    if global_motion <= static_thresh:
        motion_conf = 1.0
    elif global_motion >= review_thresh:
        motion_conf = 0.0
    else:
        motion_conf = 1.0 - ((global_motion - static_thresh) / (review_thresh - static_thresh))

    # 2. Spatial zone score: 0 active zones -> 1.0, >= active_zone_ratio * 1.5 -> 0.0
    zone_thresh = cfg.active_zone_ratio * 1.5
    if zone_ratio <= 0.0:
        zone_conf = 1.0
    elif zone_ratio >= zone_thresh:
        zone_conf = 0.0
    else:
        zone_conf = 1.0 - (zone_ratio / zone_thresh)

    # Pure visual motion: 60% global motion, 40% spatial zone distribution
    conf = (motion_conf * 0.60) + (zone_conf * 0.40)

    # Penalty for very short clips (< 10s) to guard against transient freeze-frames
    if 0 < duration < 5.0:
        conf *= 0.70
    elif 0 < duration < 10.0:
        conf *= 0.85

    return float(np.clip(conf, 0.0, 1.0))


def evaluate_decision(conf: float, cfg: SensitivityConfig) -> Tuple[str, str]:
    """Evaluates final confidence against preset thresholds."""
    if conf >= cfg.confidence_static:
        decision = "static"
    elif conf >= cfg.confidence_review:
        decision = "review"
    else:
        decision = "dynamic"

    return decision, f"{conf:.2f}"


def detect_video(
    video_path: Path,
    cfg: Any = "medium",
    debug: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Full pipeline to probe, sample, and classify a single video item matching StaticSorter.
    """
    video_path = Path(video_path)
    if isinstance(cfg, str):
        cfg = SENSITIVITY_PRESETS.get(cfg, SENSITIVITY_PRESETS["medium"])
    elif isinstance(cfg, dict) and not isinstance(cfg, SensitivityConfig):
        cfg = SensitivityConfig(
            global_motion_static=float(cfg.get("global_motion_static", 4.5)),
            global_motion_review=float(cfg.get("global_motion_review", 12.0)),
            active_zone_ratio=float(cfg.get("active_zone_ratio", 0.25)),
            confidence_static=float(cfg.get("confidence_static", 0.65)),
            confidence_review=float(cfg.get("confidence_review", 0.40)),
        )

    result: Dict[str, Any] = {
        "filename": str(video_path),
        "asset_id": "",
        "decision": "error: unknown",
        "final_confidence": "0.0",
        "global_motion_score": "0.0",
        "active_zone_ratio": "0.0",
        "max_zone_motion": "0.0",
        "heuristic_score": "0.0",
        "duration_s": "0.0",
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frames_sampled": 0,
        "error": "",
    }

    try:
        from immich_static.exif import probe_video
        meta = probe_video(video_path)
        result["duration_s"] = str(round(meta.duration_s, 2))
        result["width"] = meta.width
        result["height"] = meta.height
        result["fps"] = round(meta.fps, 2)

        if meta.width == 0 and meta.height == 0:
            result["decision"] = "ignored: audio-only"
            result["error"] = "No video stream in container (audio-only file)"
            if debug:
                print(f"ℹ️  [DEBUG] Audio-only file (no video stream): {video_path.name}")
            return result

        if meta.duration_s <= 0 and meta.width == 0:
            result["decision"] = "error: probe failed"
            result["error"] = "Failed to probe video stream/duration"
            if debug:
                print(f"⚠️  [DEBUG] Probe failed for {video_path}")
            return result

        frames = extract_sparse_frames(video_path, meta.duration_s)
        result["frames_sampled"] = len(frames)

        if len(frames) < 2:
            result["decision"] = "error: failed to extract frames"
            result["error"] = "Insufficient frames extracted"
            if debug:
                print(f"⚠️  [DEBUG] Insufficient frames ({len(frames)}) extracted from {video_path} (duration: {meta.duration_s}s)")
            return result

        g_motion = compute_global_motion(frames)
        z_ratio, max_z = compute_spatial_zone_ratio(frames)
        conf = compute_confidence(g_motion, z_ratio, cfg=cfg, duration=meta.duration_s)
        decision, conf_str = evaluate_decision(conf, cfg)

        result["global_motion_score"] = f"{g_motion:.2f}"
        result["active_zone_ratio"] = f"{z_ratio:.2f}"
        result["max_zone_motion"] = f"{max_z:.2f}"
        result["heuristic_score"] = "0.00"
        result["final_confidence"] = conf_str
        result["decision"] = decision

    except Exception as e:
        result["decision"] = f"error: {e}"
        result["error"] = str(e)

    return result


# ─────────────────────────────────────────────
# SHARPNESS EVALUATION & FRAME EXTRACTION
# ─────────────────────────────────────────────

def calculate_sharpness(image: Any) -> float:
    """Calculates Laplacian variance to measure sharpness."""
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pick_best_frame(frames: List[Any]) -> int:
    """
    Selects the index of the highest quality frame:
    - High sharpness (Laplacian variance)
    - Low local motion relative to neighbors (calmness)
    - Excludes first and last frames to avoid fade-in/fade-out artifacts.
    Composite score: 60% calmness, 40% sharpness.
    """
    import cv2
    import numpy as np

    n = len(frames)
    if n <= 1:
        return 0
    if n == 2:
        return 0

    motion = np.zeros(n, dtype=np.float64)
    sharp = np.zeros(n, dtype=np.float64)

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f for f in frames]

    for i in range(n):
        sharp[i] = cv2.Laplacian(grays[i].astype(np.float64), cv2.CV_64F).var()

    for i in range(1, n - 1):
        gp = grays[i - 1].astype(np.float64)
        gc = grays[i].astype(np.float64)
        gn = grays[i + 1].astype(np.float64)
        motion[i] = (np.mean(np.abs(gc - gp)) + np.mean(np.abs(gc - gn))) / 2.0

    # Normalize strictly across candidates [1, n-2] to avoid baseline skew
    cand_motion = motion[1:-1]
    cand_sharp = sharp[1:-1]

    def norm(a: np.ndarray) -> np.ndarray:
        mn, mx = a.min(), a.max()
        return np.zeros_like(a) if mx == mn else (a - mn) / (mx - mn)

    norm_motion = norm(cand_motion)
    norm_sharp = norm(cand_sharp)

    cand_score = (1.0 - norm_motion) * 0.6 + norm_sharp * 0.4
    return int(np.argmax(cand_score)) + 1


def extract_one_frame(
    video_path: Path,
    out_path: Path,
    fmt: str = "jpg",
    quality: int = 95,
    num_candidates: int = 12,
    tags: Optional[List[str]] = None,
    original_file_name: Optional[str] = None,
    date_hint: Optional[str] = None,
    metadata_hint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extracts the highest quality calm frame from a video and saves to disk
    with complete EXIF metadata, GPS, Adobe XMP packets, and synchronized timestamps.
    """
    import cv2
    import numpy as np
    from immich_static.exif import probe_video, save_image_with_metadata

    video_path = Path(video_path)
    out_path = Path(out_path)
    meta = probe_video(
        video_path,
        original_file_name=original_file_name,
        date_hint=date_hint,
        metadata_hint=metadata_hint,
    )
    if meta.width == 0 and meta.height == 0:
        return {"file": video_path.name, "status": "error: no video stream found", "output": ""}

    duration = meta.duration_s if meta.duration_s > 0 else 1.0
    n_samples = max(4, min(20, num_candidates))

    frames = []
    sampled_ts = []

    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

            if total_frames >= 4:
                indices = [
                    max(0, min(total_frames - 1, int(total_frames * (i + 0.5) / n_samples)))
                    for i in range(n_samples)
                ]
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frames.append(frame)
                        sampled_ts.append(round(idx / fps, 3))

            if len(frames) < 3:
                frames = []
                sampled_ts = []
                margin = max(0.04, min(0.5, duration * 0.05))
                t_start = margin
                t_end = max(t_start + 0.04, duration - margin)
                step = (t_end - t_start) / max(1, n_samples - 1)
                timestamps = [t_start + i * step for i in range(n_samples)]

                for ts in timestamps:
                    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        frame_idx = int(ts * fps)
                        if total_frames > 0:
                            frame_idx = min(total_frames - 1, frame_idx)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()

                    if ret and frame is not None:
                        frames.append(frame)
                        sampled_ts.append(ts)
        finally:
            cap.release()

    # Fallback to ffmpeg CLI pipe if OpenCV could not read sufficient frames
    if len(frames) < 3:
        frames = []
        sampled_ts = []
        for ts in timestamps:
            cmd = [
                "ffmpeg",
                "-ss", f"{ts:.3f}",
                "-i", str(video_path),
                "-vframes", "1",
                "-f", "image2pipe",
                "-vcodec", "png",
                "-"
            ]
            res = subprocess.run(cmd, capture_output=True, check=False)
            if res.returncode == 0 and res.stdout:
                nparr = np.frombuffer(res.stdout, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    frames.append(img)
                    sampled_ts.append(ts)

    if not frames:
        return {"file": video_path.name, "status": "error: no frame extracted", "output": ""}

    best_idx = pick_best_frame(frames)
    best_frame = frames[best_idx]
    best_ts = sampled_ts[best_idx]
    sharpness_score = calculate_sharpness(best_frame)

    ok = save_image_with_metadata(
        frame_bgr=best_frame,
        output_path=out_path,
        source_video_path=video_path,
        meta=meta,
        fmt=fmt,
        quality=quality,
        tags=tags,
    )

    if not ok:
        return {"file": video_path.name, "status": "error: failed to save frame with metadata", "output": ""}

    return {
        "file": video_path.name,
        "status": "ok",
        "best_frame_idx": best_idx,
        "total_evaluated": len(frames),
        "sharpness": round(sharpness_score, 2),
        "timestamp_s": round(best_ts, 2),
        "output": str(out_path),
    }


# ─────────────────────────────────────────────
# IMMICH PATH RESOLUTION
# ─────────────────────────────────────────────

def resolve_local_video_path(original_path: str, library_root: Path) -> Optional[Path]:
    """
    Resolves Immich server originalPath to the corresponding file on the host.
    Handles various Immich mount directory structures seamlessly.
    """
    if not original_path:
        return None

    # 1. Direct absolute path
    direct = Path(original_path)
    if direct.is_file():
        return direct

    # 2. Check directly under library_root
    clean_path = original_path.lstrip("/").replace("\\", "/")
    p1 = library_root / clean_path
    if p1.is_file():
        return p1

    # 3. Strip common Immich container prefixes
    prefixes_to_strip = [
        "usr/src/app/upload/",
        "upload/",
        "data/",
        "library/",
    ]
    for prefix in prefixes_to_strip:
        if clean_path.startswith(prefix):
            sub = clean_path[len(prefix):]
            cand = library_root / sub
            if cand.is_file():
                return cand
            cand2 = library_root.parent / clean_path
            if cand2.is_file():
                return cand2

    # 4. Check relative to parent directory of library_root
    cand3 = library_root.parent / clean_path
    if cand3.is_file():
        return cand3

    # 5. Check by trailing subpaths (e.g. userId/year/date/filename)
    parts = Path(clean_path).parts
    for i in range(1, len(parts)):
        subpath = Path(*parts[i:])
        cand = library_root / subpath
        if cand.is_file():
            return cand
        cand_parent = library_root.parent / subpath
        if cand_parent.is_file():
            return cand_parent

    return None


# ─────────────────────────────────────────────
# SQLITE RESUMABLE CHECKPOINT MANAGER
# ─────────────────────────────────────────────

class Checkpoint:
    """
    SQLite-backed checkpoint with thread-safe WAL mode.
    Maintains fast in-memory status cache for quick lookups across 60k+ items.
    Supports lookups by asset_id or filename.
    """
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache_by_filename: Dict[str, Dict[str, Any]] = {}
        self._cache_by_asset_id: Dict[str, Dict[str, Any]] = {}
        self._init_db()
        self._load_cache()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=60.0)
        conn.execute("PRAGMA busy_timeout = 60000")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_conn() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    asset_id TEXT PRIMARY KEY,
                    filename TEXT,
                    original_file_name TEXT,
                    decision TEXT,
                    final_confidence TEXT,
                    global_motion_score TEXT,
                    active_zone_ratio TEXT,
                    max_zone_motion TEXT,
                    heuristic_score TEXT,
                    duration_s TEXT,
                    width INTEGER,
                    height INTEGER,
                    fps REAL,
                    frames_sampled INTEGER,
                    error TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            extra_cols = [
                ("original_file_name", "TEXT"),
                ("heuristic_score", "TEXT"),
                ("date_time_original", "TEXT"),
                ("file_created_at", "TEXT"),
                ("latitude", "REAL"),
                ("longitude", "REAL"),
                ("altitude", "REAL"),
                ("camera_make", "TEXT"),
                ("camera_model", "TEXT"),
                ("synced_tag", "TEXT"),
                ("synced_album", "TEXT"),
                ("synced_at", "TEXT"),
                ("user_curated", "INTEGER DEFAULT 0"),
            ]
            for col, col_type in extra_cols:
                try:
                    conn.execute(f"ALTER TABLE videos ADD COLUMN {col} {col_type}")
                except Exception:
                    pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_filename ON videos(filename)")
            conn.commit()

    def _load_cache(self):
        with self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM videos")
            for row in cur.fetchall():
                d = dict(row)
                aid = d.get("asset_id")
                fn = d.get("filename")
                if aid:
                    self._cache_by_asset_id[aid] = d
                if fn:
                    self._cache_by_filename[fn] = d

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached record by asset_id or filename."""
        return self._cache_by_asset_id.get(key) or self._cache_by_filename.get(key)

    def is_done(self, key: str) -> bool:
        row = self._cache_by_asset_id.get(key) or self._cache_by_filename.get(key)
        if not row:
            return False
        dec = row.get("decision", "")
        return dec in ("static", "dynamic", "review")

    def is_error(self, key: str) -> bool:
        row = self._cache_by_asset_id.get(key) or self._cache_by_filename.get(key)
        if not row:
            return False
        return str(row.get("decision", "")).startswith("error")

    def _build_video_row_tuple(self, asset_id_or_filename: str, data: Dict[str, Any]) -> Tuple:
        aid = data.get("asset_id") or asset_id_or_filename
        fn = data.get("filename") or asset_id_or_filename
        orig_name = data.get("original_file_name") or Path(fn).name
        return (
            aid,
            fn,
            orig_name,
            data.get("decision", ""),
            str(data.get("final_confidence", "")),
            str(data.get("global_motion_score", "")),
            str(data.get("active_zone_ratio", "")),
            str(data.get("max_zone_motion", "")),
            str(data.get("heuristic_score", "")),
            str(data.get("duration_s", "")),
            int(data.get("width") or 0),
            int(data.get("height") or 0),
            float(data.get("fps") or 0.0),
            int(data.get("frames_sampled") or 0),
            data.get("error", ""),
            str(data.get("date_time_original") or ""),
            str(data.get("file_created_at") or ""),
            float(data["latitude"]) if data.get("latitude") is not None else None,
            float(data["longitude"]) if data.get("longitude") is not None else None,
            float(data["altitude"]) if data.get("altitude") is not None else None,
            str(data.get("camera_make") or ""),
            str(data.get("camera_model") or ""),
            str(data.get("synced_tag")) if data.get("synced_tag") is not None else None,
            str(data.get("synced_at")) if data.get("synced_at") is not None else None,
            int(data.get("user_curated") or 0),
        )

    def record(self, asset_id_or_filename: str, data: Dict[str, Any]):
        aid = data.get("asset_id") or asset_id_or_filename
        fn = data.get("filename") or asset_id_or_filename
        params = self._build_video_row_tuple(asset_id_or_filename, data)
        with self._lock:
            self._cache_by_asset_id[aid] = data
            self._cache_by_filename[fn] = data
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO videos (
                        asset_id, filename, original_file_name, decision, final_confidence,
                        global_motion_score, active_zone_ratio, max_zone_motion, heuristic_score,
                        duration_s, width, height, fps, frames_sampled, error,
                        date_time_original, file_created_at, latitude, longitude, altitude,
                        camera_make, camera_model, synced_tag, synced_at, user_curated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, params)
                conn.commit()

    def record_batch(self, items: List[Tuple[str, Dict[str, Any]]]):
        if not items:
            return
        row_params = []
        with self._lock:
            for key, data in items:
                aid = data.get("asset_id") or key
                fn = data.get("filename") or key
                self._cache_by_asset_id[aid] = data
                self._cache_by_filename[fn] = data
                row_params.append(self._build_video_row_tuple(key, data))

            with self._get_conn() as conn:
                conn.executemany("""
                    INSERT OR REPLACE INTO videos (
                        asset_id, filename, original_file_name, decision, final_confidence,
                        global_motion_score, active_zone_ratio, max_zone_motion, heuristic_score,
                        duration_s, width, height, fps, frames_sampled, error,
                        date_time_original, file_created_at, latitude, longitude, altitude,
                        camera_make, camera_model, synced_tag, synced_at, user_curated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, row_params)
                conn.commit()

    def update_decision(
        self,
        asset_id_or_filename: str,
        new_decision: str,
        new_confidence: Optional[str] = None,
        user_curated: Optional[int] = None,
    ) -> bool:
        """Updates the decision and optional confidence for a video in both memory cache and SQLite database."""
        with self._lock:
            row = self._cache_by_asset_id.get(asset_id_or_filename) or self._cache_by_filename.get(asset_id_or_filename)
            if not row:
                return False
            row["decision"] = new_decision
            if new_confidence is not None:
                row["final_confidence"] = str(new_confidence)
            if user_curated is not None:
                row["user_curated"] = user_curated
            aid = row.get("asset_id")
            fn = row.get("filename")
            if aid:
                self._cache_by_asset_id[aid] = row
            if fn:
                self._cache_by_filename[fn] = row
            with self._get_conn() as conn:
                if new_confidence is not None and user_curated is not None:
                    conn.execute(
                        "UPDATE videos SET decision = ?, final_confidence = ?, user_curated = ? WHERE asset_id = ? OR filename = ?",
                        (new_decision, str(new_confidence), user_curated, aid or asset_id_or_filename, fn or asset_id_or_filename)
                    )
                elif new_confidence is not None:
                    conn.execute(
                        "UPDATE videos SET decision = ?, final_confidence = ? WHERE asset_id = ? OR filename = ?",
                        (new_decision, str(new_confidence), aid or asset_id_or_filename, fn or asset_id_or_filename)
                    )
                elif user_curated is not None:
                    conn.execute(
                        "UPDATE videos SET decision = ?, user_curated = ? WHERE asset_id = ? OR filename = ?",
                        (new_decision, user_curated, aid or asset_id_or_filename, fn or asset_id_or_filename)
                    )
                else:
                    conn.execute(
                        "UPDATE videos SET decision = ? WHERE asset_id = ? OR filename = ?",
                        (new_decision, aid or asset_id_or_filename, fn or asset_id_or_filename)
                    )
                conn.commit()
            return True

    def update_decisions_batch(
        self,
        items: List[Tuple[str, str, Optional[str]]],
        user_curated: Optional[int] = None,
    ):
        """Batch updates decisions and optional confidences in a single fast transaction."""
        if not items:
            return
        params_by_aid = []
        params_by_fn = []
        with self._lock:
            for key, new_decision, new_confidence in items:
                row = self._cache_by_asset_id.get(key) or self._cache_by_filename.get(key)
                if row:
                    row["decision"] = new_decision
                    if new_confidence is not None:
                        row["final_confidence"] = str(new_confidence)
                    if user_curated is not None:
                        row["user_curated"] = user_curated
                    aid = row.get("asset_id")
                    fn = row.get("filename")
                    if aid:
                        self._cache_by_asset_id[aid] = row
                    if fn:
                        self._cache_by_filename[fn] = row

                # Choose primary key if known for instant O(1) indexed update
                aid = (row.get("asset_id") if row else None) or (key if key in self._cache_by_asset_id else None)
                conf_val = str(new_confidence) if new_confidence is not None else (row.get("final_confidence", "") if row else "")
                if aid:
                    if user_curated is not None:
                        params_by_aid.append((new_decision, conf_val, user_curated, aid))
                    else:
                        params_by_aid.append((new_decision, conf_val, aid))
                else:
                    if user_curated is not None:
                        params_by_fn.append((new_decision, conf_val, user_curated, key))
                    else:
                        params_by_fn.append((new_decision, conf_val, key))

            with self._get_conn() as conn:
                if user_curated is not None:
                    if params_by_aid:
                        conn.executemany("UPDATE videos SET decision = ?, final_confidence = ?, user_curated = ? WHERE asset_id = ?", params_by_aid)
                    if params_by_fn:
                        conn.executemany("UPDATE videos SET decision = ?, final_confidence = ?, user_curated = ? WHERE filename = ?", params_by_fn)
                else:
                    if params_by_aid:
                        conn.executemany("UPDATE videos SET decision = ?, final_confidence = ? WHERE asset_id = ?", params_by_aid)
                    if params_by_fn:
                        conn.executemany("UPDATE videos SET decision = ?, final_confidence = ? WHERE filename = ?", params_by_fn)
                conn.commit()

    def update_sync_status(
        self,
        asset_id_or_filename: str,
        synced_album: Optional[str],
        synced_at: Optional[str] = None,
    ) -> bool:
        """Updates the synced_album and synced_at for an asset in cache and database."""
        actual_time = synced_at or datetime.now().isoformat()
        with self._lock:
            row = self._cache_by_asset_id.get(asset_id_or_filename) or self._cache_by_filename.get(asset_id_or_filename)
            aid = row.get("asset_id") if row else (asset_id_or_filename if asset_id_or_filename in self._cache_by_asset_id else None)
            if row:
                row["synced_album"] = synced_album
                row["synced_at"] = actual_time
            with self._get_conn() as conn:
                if aid:
                    conn.execute("UPDATE videos SET synced_album = ?, synced_at = ? WHERE asset_id = ?", (synced_album, actual_time, aid))
                else:
                    conn.execute("UPDATE videos SET synced_album = ?, synced_at = ? WHERE filename = ?", (synced_album, actual_time, asset_id_or_filename))
                conn.commit()
            return True

    def update_sync_status_batch(
        self,
        items: List[Tuple[str, Optional[str], Optional[str]]],
    ):
        """Batch updates synced_album and synced_at in a single fast transaction."""
        if not items:
            return
        now_str = datetime.now().isoformat()
        params_by_aid = []
        params_by_fn = []
        with self._lock:
            for key, album, s_at in items:
                actual_time = s_at or now_str
                row = self._cache_by_asset_id.get(key) or self._cache_by_filename.get(key)
                if row:
                    row["synced_album"] = album
                    row["synced_at"] = actual_time
                aid = (row.get("asset_id") if row else None) or (key if key in self._cache_by_asset_id else None)
                if aid:
                    params_by_aid.append((album, actual_time, aid))
                else:
                    params_by_fn.append((album, actual_time, key))

            with self._get_conn() as conn:
                if params_by_aid:
                    conn.executemany("UPDATE videos SET synced_album = ?, synced_at = ? WHERE asset_id = ?", params_by_aid)
                if params_by_fn:
                    conn.executemany("UPDATE videos SET synced_album = ?, synced_at = ? WHERE filename = ?", params_by_fn)
                conn.commit()

    def save_meta(self, **kwargs):
        with self._get_conn() as conn:
            for k, v in kwargs.items():
                conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (k, str(v)))
            conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        with self._get_conn() as conn:
            cur = conn.execute("SELECT value FROM meta WHERE key = ?", (key,))
            row = cur.fetchone()
            return row["value"] if row else None

    def all_rows(self) -> List[Dict[str, Any]]:
        with self._lock:
            with self._get_conn() as conn:
                cur = conn.execute("SELECT * FROM videos")
                return [dict(r) for r in cur.fetchall()]

    def prune_missing(self, active_asset_ids: Set[str], active_filenames: Optional[Set[str]] = None) -> int:
        """Removes entries from the checkpoint database that are neither in active_asset_ids nor active_filenames."""
        active_filenames = active_filenames or set()
        with self._lock:
            with self._get_conn() as conn:
                cur = conn.execute("SELECT asset_id, filename FROM videos")
                rows = cur.fetchall()
                to_delete = [
                    (r["asset_id"], r["filename"])
                    for r in rows
                    if r["asset_id"] not in active_asset_ids and r["filename"] not in active_filenames
                ]
                for aid, fn in to_delete:
                    conn.execute("DELETE FROM videos WHERE asset_id = ? AND filename = ?", (aid, fn))
                conn.commit()
            self._cache_by_asset_id.clear()
            self._cache_by_filename.clear()
            self._load_cache()
            return len(to_delete)

    def clear(self):
        with self._lock:
            self._cache_by_asset_id.clear()
            self._cache_by_filename.clear()
            with self._get_conn() as conn:
                conn.execute("DELETE FROM videos")
                conn.execute("DELETE FROM meta")
                conn.commit()

    def wait_for_writes(self):
        pass

    def flush_and_stop(self):
        pass
