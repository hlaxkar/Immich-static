"""
core.py — Shared engine for Immich static video detection and frame extraction.
"""

import os
import sys
import json
import shutil
import sqlite3
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ─────────────────────────────────────────────
# AUTO-LOAD .ENV FILE IF PRESENT
# ─────────────────────────────────────────────

def _load_dotenv():
    env_file = Path(".env")
    if not env_file.exists():
        env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v

_load_dotenv()

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

LOG_FIELDS = [
    "filename",
    "original_file_name",
    "asset_id",
    "decision",
    "final_confidence",
    "global_motion_score",
    "active_zone_ratio",
    "max_zone_motion",
    "duration_s",
    "width",
    "height",
    "fps",
    "frames_sampled",
    "error",
]

SENSITIVITY_PRESETS = {
    "low": {
        "static_max_motion": 0.015,
        "static_max_active_zones": 0.06,
        "review_max_motion": 0.040,
        "review_max_active_zones": 0.15,
    },
    "medium": {
        "static_max_motion": 0.028,
        "static_max_active_zones": 0.12,
        "review_max_motion": 0.065,
        "review_max_active_zones": 0.25,
    },
    "high": {
        "static_max_motion": 0.045,
        "static_max_active_zones": 0.18,
        "review_max_motion": 0.095,
        "review_max_active_zones": 0.35,
    },
}


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


def extract_sparse_frames(
    video_path: Path,
    duration: float,
    num_samples: int = 10,
    max_dimension: int = 360
) -> List[Any]:
    """
    Extracts sparse downscaled frames for rapid motion analysis.
    Uses OpenCV / FFmpeg to read timestamps evenly spaced across video.
    """
    import cv2
    import numpy as np

    if duration <= 0:
        duration = 1.0

    timestamps = [duration * (i + 0.5) / num_samples for i in range(num_samples)]
    frames = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                frame_idx = int(ts * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

            if ret and frame is not None:
                h, w = frame.shape[:2]
                if max(h, w) > max_dimension:
                    scale = max_dimension / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                frames.append(frame)
    finally:
        cap.release()

    if len(frames) < 2:
        frames = _ffmpeg_extract_sparse(video_path, timestamps, max_dimension)

    return frames


def _ffmpeg_extract_sparse(video_path: Path, timestamps: List[float], max_dimension: int = 360) -> List[Any]:
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
            "-vf", f"scale='min({max_dimension},iw)':-2",
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
# MOTION DETECTION & GRID ANALYSIS
# ─────────────────────────────────────────────

def analyze_motion(frames: List[Any], grid_size: int = 4) -> Tuple[float, float, float]:
    """
    Analyzes motion between consecutive downscaled frames.
    Returns:
      (global_motion_score, active_zone_ratio, max_zone_motion)
    """
    import cv2
    import numpy as np

    if len(frames) < 2:
        return 0.0, 0.0, 0.0

    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    h, w = gray_frames[0].shape

    cell_h = h // grid_size
    cell_w = w // grid_size

    diff_scores = []
    zone_motions = np.zeros((grid_size, grid_size), dtype=np.float32)

    for i in range(len(gray_frames) - 1):
        prev = gray_frames[i]
        curr = gray_frames[i + 1]

        diff = cv2.absdiff(curr, prev) / 255.0
        global_diff = float(np.mean(diff))
        diff_scores.append(global_diff)

        for r in range(grid_size):
            for c in range(grid_size):
                zone = diff[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                zone_motions[r, c] += float(np.mean(zone))

    num_transitions = len(gray_frames) - 1
    if num_transitions > 0:
        zone_motions /= num_transitions

    global_motion = float(np.mean(diff_scores))
    max_zone_motion = float(np.max(zone_motions))

    zone_active_threshold = 0.035
    active_zones = np.sum(zone_motions > zone_active_threshold)
    active_zone_ratio = float(active_zones / (grid_size * grid_size))

    return round(global_motion, 4), round(active_zone_ratio, 3), round(max_zone_motion, 4)


def detect_video(video_path: Path, thresholds: Dict[str, float], debug: bool = False) -> Dict[str, Any]:
    """
    Performs full detection and classification on a single video file.
    """
    result: Dict[str, Any] = {
        "filename": str(video_path),
        "asset_id": "",
        "decision": "error: unknown",
        "final_confidence": "0.0",
        "global_motion_score": "0.0",
        "active_zone_ratio": "0.0",
        "max_zone_motion": "0.0",
        "duration_s": "0.0",
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frames_sampled": 0,
        "error": "",
    }

    try:
        meta = get_video_metadata(video_path)
        result["duration_s"] = str(meta["duration"])
        result["width"] = meta["width"]
        result["height"] = meta["height"]
        result["fps"] = meta["fps"]

        num_samples = 10 if meta["duration"] <= 60 else min(15, int(meta["duration"] / 4))
        frames = extract_sparse_frames(video_path, meta["duration"], num_samples=num_samples)
        result["frames_sampled"] = len(frames)

        if len(frames) < 2:
            result["decision"] = "error: failed to extract frames"
            result["error"] = "Insufficient frames extracted"
            return result

        global_motion, active_ratio, max_zone = analyze_motion(frames)
        result["global_motion_score"] = f"{global_motion:.4f}"
        result["active_zone_ratio"] = f"{active_ratio:.3f}"
        result["max_zone_motion"] = f"{max_zone:.4f}"

        static_max_m = thresholds["static_max_motion"]
        static_max_z = thresholds["static_max_active_zones"]
        review_max_m = thresholds["review_max_motion"]
        review_max_z = thresholds["review_max_active_zones"]

        if global_motion <= static_max_m and active_ratio <= static_max_z:
            result["decision"] = "static"
            conf = max(0.6, 1.0 - (global_motion / static_max_m) * 0.4)
            result["final_confidence"] = f"{conf:.2f}"
        elif global_motion <= review_max_m or active_ratio <= review_max_z:
            result["decision"] = "review"
            conf = 0.5 + 0.3 * (1.0 - min(1.0, global_motion / review_max_m))
            result["final_confidence"] = f"{conf:.2f}"
        else:
            result["decision"] = "dynamic"
            conf = min(0.99, 0.6 + (global_motion / (review_max_m * 2)) * 0.4)
            result["final_confidence"] = f"{conf:.2f}"

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


def extract_one_frame(
    video_path: Path,
    out_path: Path,
    fmt: str = "jpg",
    quality: int = 95,
    num_candidates: int = 8
) -> Dict[str, Any]:
    """
    Extracts the highest sharpness frame from a video and saves to disk.
    """
    import cv2

    meta = get_video_metadata(video_path)
    duration = meta["duration"] or 1.0
    timestamps = [duration * (i + 0.5) / num_candidates for i in range(num_candidates)]

    best_frame = None
    best_score = -1.0
    best_ts = timestamps[0]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
                ret, frame = cap.read()

            if ret and frame is not None:
                score = calculate_sharpness(frame)
                if score > best_score:
                    best_score = score
                    best_frame = frame
                    best_ts = ts
    finally:
        cap.release()

    if best_frame is None:
        return {"file": video_path.name, "status": "error: no frame extracted", "output": ""}

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use ffmpeg with -map_metadata 0 to export frame with EXIF/metadata preserved
    saved_with_ffmpeg = False
    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{best_ts:.3f}",
            "-i", str(video_path),
            "-vframes", "1",
            "-map_metadata", "0",
            str(out_path)
        ]
        res = subprocess.run(cmd, capture_output=True, check=False)
        if res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            saved_with_ffmpeg = True
    except Exception:
        pass

    if not saved_with_ffmpeg:
        if fmt.lower() == "png":
            cv2.imwrite(str(out_path), best_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:
            cv2.imwrite(str(out_path), best_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

    # Preserve file access/modification timestamps
    try:
        stat = video_path.stat()
        os.utime(out_path, (stat.st_atime, stat.st_mtime))
    except Exception:
        pass

    return {
        "file": video_path.name,
        "status": "ok",
        "sharpness": round(best_score, 2),
        "timestamp_s": round(best_ts, 2),
        "output": str(out_path),
    }


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
        self.db_path = db_path
        self._lock = threading.Lock()
        self._cache_by_filename: Dict[str, Dict[str, Any]] = {}
        self._cache_by_asset_id: Dict[str, Dict[str, Any]] = {}
        self._init_db()
        self._load_cache()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
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
                    duration_s TEXT,
                    width INTEGER,
                    height INTEGER,
                    fps REAL,
                    frames_sampled INTEGER,
                    error TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Migration check: add column if existing db didn't have it
            try:
                conn.execute("ALTER TABLE videos ADD COLUMN original_file_name TEXT")
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

    def record(self, asset_id_or_filename: str, data: Dict[str, Any]):
        aid = data.get("asset_id") or asset_id_or_filename
        fn = data.get("filename") or asset_id_or_filename
        orig_name = data.get("original_file_name") or Path(fn).name
        with self._lock:
            self._cache_by_asset_id[aid] = data
            self._cache_by_filename[fn] = data
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO videos (
                        asset_id, filename, original_file_name, decision, final_confidence,
                        global_motion_score, active_zone_ratio, max_zone_motion,
                        duration_s, width, height, fps, frames_sampled, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    aid,
                    fn,
                    orig_name,
                    data.get("decision", ""),
                    str(data.get("final_confidence", "")),
                    str(data.get("global_motion_score", "")),
                    str(data.get("active_zone_ratio", "")),
                    str(data.get("max_zone_motion", "")),
                    str(data.get("duration_s", "")),
                    int(data.get("width") or 0),
                    int(data.get("height") or 0),
                    float(data.get("fps") or 0.0),
                    int(data.get("frames_sampled") or 0),
                    data.get("error", ""),
                ))
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
            return list(self._cache_by_asset_id.values())

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
