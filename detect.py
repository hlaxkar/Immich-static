#!/usr/bin/env python3
"""
detect.py — API-First, Zero-Download Static Video Detector for Immich.

Workflow:
1. Connects to Immich REST API using your API Key (100% user-scoped).
2. Fetches video asset catalog with exact Asset IDs and originalPaths.
3. Resolves files directly on your local mounted disk (zero download).
4. Analyzes multi-frame motion and records classifications into SQLite.
5. Automatically syncs tags [Static, Review, Dynamic] and albums to Immich.

Usage:
    python detect.py [options]

Options:
    --library PATH          Path to mounted Immich library (or set IMMICH_LIBRARY_PATH in .env)
    --api-url URL           Immich API URL (or set IMMICH_API_URL in .env)
    --api-key KEY           Immich API Key (or set IMMICH_API_KEY in .env)
    --sensitivity LEVEL     low | medium | high  (default: medium)
    --workers N             Parallel workers (default: auto based on CPU)
    --db PATH               Path to SQLite checkpoint (default: ./detection_checkpoint.sqlite)
    --no-sync               Skip automatic syncing of tags and albums after detection
    --export-csv PATH       Optional: Export results to a CSV spreadsheet
    --fresh                 Ignore checkpoint, re-detect everything
    --report                Print detailed motion scores breakdown
    --debug                 Print ffprobe/ffmpeg errors for failed videos
"""

import argparse
import csv
import os
import signal
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import atexit

try:
    import termios
except ImportError:
    termios = None

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from core import (
    ENV, SENSITIVITY_PRESETS, CHECKPOINT_FILENAME, LOG_FIELDS,
    Checkpoint, detect_video, resolve_local_video_path,
)
from immich_sync import ImmichClient, sync_immich

# ─────────────────────────────────────────────
# TERMINAL STATE PROTECTION
# ─────────────────────────────────────────────

_saved_term_attrs = None

def _save_terminal_state():
    global _saved_term_attrs
    if termios is None:
        return
    try:
        _saved_term_attrs = termios.tcgetattr(sys.stdin.fileno())
    except (termios.error, ValueError, OSError):
        pass

def _restore_terminal_state():
    if _saved_term_attrs is not None and termios is not None:
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN,
                              _saved_term_attrs)
        except (termios.error, ValueError, OSError):
            pass

# ─────────────────────────────────────────────
# INTERRUPT HANDLING
# ─────────────────────────────────────────────

import threading

_interrupt_event = threading.Event()
_hard_stop_event = threading.Event()
_sig_count       = 0
_sig_lock        = threading.Lock()

def _handle_sigint(sig, frame):
    global _sig_count
    with _sig_lock:
        _sig_count += 1
        if _sig_count == 1:
            print(
                "\n\n⚠️  Interrupt received — finishing in-flight videos then saving checkpoint...\n"
                "   Press Ctrl+C again to force quit immediately.\n"
            )
            _interrupt_event.set()
        else:
            print("\n🛑 Force quit.\n")
            _hard_stop_event.set()
            _restore_terminal_state()
            sys.exit(1)

# ─────────────────────────────────────────────
# PROGRESS BAR
# ─────────────────────────────────────────────

class FallbackBar:
    def __init__(self, total, desc=""):
        self.total = total
        self.desc  = desc
        self.n     = 0

    def update(self, n=1):
        self.n += n
        pct = int(self.n / self.total * 40) if self.total else 0
        bar = "█" * pct + "░" * (40 - pct)
        print(f"\r{self.desc}: [{bar}] {self.n}/{self.total}", end="", flush=True)

    def set_postfix(self, d):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        print()

def make_bar(total, desc):
    if HAS_TQDM:
        return tqdm(total=total, desc=desc, unit="video")
    return FallbackBar(total=total, desc=desc)

# ─────────────────────────────────────────────
# REPORT & SUMMARY
# ─────────────────────────────────────────────

def print_report(rows: list[dict]):
    decisions = ["static", "review", "dynamic"]
    grouped   = {d: [] for d in decisions}
    errors    = []

    for r in rows:
        d = r.get("decision", "")
        if d in grouped:
            grouped[d].append(r)
        elif d.startswith("error"):
            errors.append(r)

    for d in decisions:
        group = grouped[d]
        if not group:
            continue
        group.sort(key=lambda r: float(r.get("final_confidence") or 0), reverse=True)
        label = {"static": "🖼  STATIC", "review": "🔍 REVIEW", "dynamic": "🎥 DYNAMIC"}[d]
        print(f"\n{label} ({len(group)})")
        print(f"  {'filename':<50} {'conf':>6}  {'motion':>7}  {'zones':>6}  {'dur':>6}")
        print(f"  {'─'*50} {'─'*6}  {'─'*7}  {'─'*6}  {'─'*6}")
        for r in group[:30]:
            print(
                f"  {r.get('filename','')[-50:]:<50} "
                f"{r.get('final_confidence','?'):>6}  "
                f"{r.get('global_motion_score','?'):>7}  "
                f"{r.get('active_zone_ratio','?'):>6}  "
                f"{r.get('duration_s','?'):>6}s"
            )
        if len(group) > 30:
            print(f"  ... and {len(group) - 30} more")

    if errors:
        print(f"\n❌ ERRORS ({len(errors)})")
        for r in errors[:20]:
            print(f"  {r.get('filename','?')} — {r.get('decision','?')}")


def print_summary(rows: list[dict], interrupted: bool, total_time: float = 0.0):
    counts: dict[str, int] = {}
    for r in rows:
        d = r.get("decision", "unknown")
        counts[d] = counts.get(d, 0) + 1

    print("\n── Detection Summary ─────────────────────────")
    labels = {
        "static":  "🖼  Static   (Ready for extraction/album)",
        "review":  "🔍 Review   (Borderline motion)",
        "dynamic": "🎥 Dynamic  (Active motion video)",
    }
    for d in ("static", "review", "dynamic"):
        if d in counts:
            print(f"  {labels[d]:<42} : {counts[d]:>5}")
    errors = sum(v for k, v in counts.items() if k.startswith("error"))
    if errors:
        print(f"  ❌ Errors                                 : {errors:>5}")
    print(f"  {'─'*50}")
    print(f"  Total Processed                           : {len(rows):>5}")

    def fmt_time(seconds: float) -> str:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0: return f"{int(h)}h {int(m)}m {s:.1f}s"
        if m > 0: return f"{int(m)}m {s:.1f}s"
        return f"{s:.2f}s"

    print(f"  Total Scan Time                           : {fmt_time(total_time):>5}")
    print(f"  {'─'*50}")

    if interrupted:
        print("\n⚠️  Interrupted — checkpoint saved. Re-run anytime to resume.")
    print()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="API-First, Zero-Download Static Video Detector for Immich.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("folder", nargs="?", default=os.environ.get("IMMICH_LIBRARY_PATH"),
                   help="Path to Immich library mount on host (or set IMMICH_LIBRARY_PATH in .env)")
    p.add_argument("--library", dest="library_path", default=None,
                   help="Alias for folder path")
    p.add_argument("--api-url", default=os.environ.get("IMMICH_API_URL"),
                   help="Immich API URL (or set IMMICH_API_URL in .env)")
    p.add_argument("--api-key", default=os.environ.get("IMMICH_API_KEY"),
                   help="Immich API Key (or set IMMICH_API_KEY in .env)")
    p.add_argument("--sensitivity",  choices=["low", "medium", "high"],
                   default=os.environ.get("SENSITIVITY", "medium"))
    p.add_argument("--workers",      type=int,
                   default=int(os.environ["WORKERS"]) if os.environ.get("WORKERS") else None,
                   help=f"Parallel workers (default: {ENV['max_workers']})")
    p.add_argument("--db", "--db-path", dest="db_path",
                   default=os.environ.get("DB_PATH", "./" + CHECKPOINT_FILENAME),
                   help="Path to SQLite checkpoint file (default: ./detection_checkpoint.sqlite)")
    p.add_argument("--sync", action="store_true",
                   help="Explicitly sync classifications to Immich Tags & Albums after detection (default: False)")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview Immich tag and album changes after scanning without modifying Immich")
    p.add_argument("--export-csv", dest="export_csv", nargs="?", const="detection_log.csv", default=None,
                   help="Optional: Export detection results to CSV (default: detection_log.csv)")
    p.add_argument("--fresh",        action="store_true",
                   help="Ignore checkpoint, re-detect everything")
    p.add_argument("--report",       action="store_true",
                   help="Print detailed confidence score breakdown")
    p.add_argument("--debug",        action="store_true",
                   help="Print ffmpeg stderr for any video that fails frame extraction")
    return p.parse_args()


def check_dependencies():
    missing = []
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            missing.append(tool)
    try:
        import cv2, numpy  # noqa: F401
    except ImportError as e:
        missing.append(str(e))

    if not missing:
        return

    print("❌ Missing dependencies:")
    for m in missing:
        print(f"   • {m}")
    print("\nInstall dependencies:")
    print("   sudo apt install ffmpeg")
    print("   pip install opencv-python-headless numpy tqdm")
    sys.exit(1)


def main():
    script_start_time = time.time()
    args = parse_args()
    _save_terminal_state()
    atexit.register(_restore_terminal_state)
    signal.signal(signal.SIGINT, _handle_sigint)

    check_dependencies()

    api_url = args.api_url or os.environ.get("IMMICH_API_URL")
    api_key = args.api_key or os.environ.get("IMMICH_API_KEY")
    library_dir_str = args.library_path or args.folder or os.environ.get("IMMICH_LIBRARY_PATH")

    if not api_url or not api_key:
        print("❌ Error: Missing Immich API configuration.")
        print("   Provide --api-url and --api-key or set IMMICH_API_URL and IMMICH_API_KEY in .env")
        sys.exit(1)

    if not library_dir_str:
        print("❌ Error: Missing Immich library mount path.")
        print("   Provide library path as argument or set IMMICH_LIBRARY_PATH in .env")
        sys.exit(1)

    library_dir = Path(library_dir_str).resolve()
    if not library_dir.is_dir():
        print(f"❌ Directory not found: {library_dir}")
        sys.exit(1)

    # 1. Connect to Immich Server
    client = ImmichClient(api_url, api_key)
    try:
        ver = client.ping()
        print(f"🌐 Connected to Immich Server (v{ver.get('major', '')}.{ver.get('minor', '')}.{ver.get('patch', '')})")
    except Exception as e:
        print(f"❌ Failed to connect to Immich server at {api_url}: {e}")
        sys.exit(1)

    # 2. Fetch User-Scoped Video Catalog via API
    immich_videos = client.get_all_video_assets()
    if not immich_videos:
        print("No video assets found in your Immich library.")
        sys.exit(0)

    # 3. Resolve Local Disk Paths
    print(f"📁 Resolving local disk paths under {library_dir}...")
    valid_targets: List[Tuple[Dict[str, Any], Path]] = []
    unresolved_count = 0

    for asset in immich_videos:
        orig_path = asset.get("originalPath") or ""
        resolved = resolve_local_video_path(orig_path, library_dir)
        if resolved and resolved.is_file():
            valid_targets.append((asset, resolved))
        else:
            unresolved_count += 1

    print(f"   • User video assets from API : {len(immich_videos)}")
    print(f"   • Located on local disk mount: {len(valid_targets)}")
    if unresolved_count > 0:
        print(f"   ⚠️  Could not locate {unresolved_count} video(s) under {library_dir}")
        print("      (Check that IMMICH_LIBRARY_PATH points to the root of your library mount).")

    if not valid_targets:
        print("❌ Error: No video files could be located on disk. Check IMMICH_LIBRARY_PATH.")
        sys.exit(1)

    thresholds = SENSITIVITY_PRESETS[args.sensitivity]
    requested  = args.workers or ENV["max_workers"]
    workers    = min(requested, ENV["max_workers"])
    ckpt_path  = Path(args.db_path).resolve()
    csv_path   = Path(args.export_csv).resolve() if args.export_csv else None

    ckpt = Checkpoint(ckpt_path)
    if args.fresh:
        ckpt.clear()
        print("🔄 Checkpoint cleared — starting fresh.\n")

    ckpt.save_meta(
        sensitivity = args.sensitivity,
        started_at  = datetime.now().isoformat(),
        library_dir = str(library_dir),
        api_url     = api_url,
    )

    # Backfill originalFileName for existing entries if needed
    for (asset, vp) in valid_targets:
        aid = asset["id"]
        cached = ckpt._cache_by_asset_id.get(aid)
        if cached and not cached.get("original_file_name"):
            cached["original_file_name"] = asset.get("originalFileName") or vp.name
            ckpt.record(aid, cached)

    # 4. Filter Assets Needing Detection
    to_detect = [
        (asset, vp) for (asset, vp) in valid_targets
        if not ckpt.is_done(asset["id"]) or ckpt.is_error(asset["id"])
    ]
    already_done = len(valid_targets) - len(to_detect)

    print(f"\n🎬 Classification Task:")
    print(f"   • Total Candidates    : {len(valid_targets)}")
    if already_done:
        print(f"   • Already Classified  : {already_done} (use --fresh to redo)")
    print(f"   • Videos to Analyse   : {len(to_detect)}")
    print(f"   • Sensitivity Profile : {args.sensitivity}")
    print(f"   • Parallel Workers    : {workers}")
    print(f"   • SQLite Checkpoint   : {ckpt_path}")
    print()

    interrupted = False

    # 5. Multi-Threaded Processing on Local Disk
    if to_detect:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for (asset, vp) in to_detect:
                if _interrupt_event.is_set():
                    break
                futures[executor.submit(detect_video, vp, thresholds, args.debug)] = (asset, vp)

            with make_bar(len(futures), "Detecting") as pbar:
                for future in as_completed(futures):
                    if _hard_stop_event.is_set():
                        break

                    (asset, vp) = futures[future]
                    aid = asset["id"]
                    try:
                        row = future.result()
                    except Exception as e:
                        row = {f: "" for f in LOG_FIELDS}
                        row["decision"] = f"error: {e}"

                    row["asset_id"] = aid
                    row["filename"] = str(vp.resolve())
                    row["original_file_name"] = asset.get("originalFileName") or vp.name
                    ckpt.record(aid, row)
                    pbar.update(1)
                    pbar.set_postfix({
                        "file":     vp.name[:20],
                        "result":   row.get("decision", "?"),
                    })

                    if _interrupt_event.is_set():
                        interrupted = True
                        break

        ckpt.wait_for_writes()

    ckpt.flush_and_stop()
    all_rows = ckpt.all_rows()

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    script_end_time = time.time()
    total_time_taken = script_end_time - script_start_time

    print_summary(all_rows, interrupted=interrupted, total_time=total_time_taken)

    if args.report:
        print_report(all_rows)

    print(f"💾 Checkpoint DB : {ckpt_path}")
    if csv_path:
        print(f"📄 Exported CSV  : {csv_path}")

    # 6. Optional Tags & Albums Sync
    if args.dry_run and not interrupted:
        print("\n🏷  [DRY-RUN] Previewing sync to Immich Tags & Albums...")
        try:
            sync_immich(
                client=client,
                ckpt=ckpt,
                dry_run=True,
                sync_tags=True,
                sync_albums=True,
                include_dynamic=True,
            )
        except Exception as e:
            print(f"⚠️  Immich dry-run preview failed: {e}")
    elif args.sync and not interrupted:
        print("\n🏷  Syncing classifications to Immich Tags & Albums...")
        try:
            sync_immich(
                client=client,
                ckpt=ckpt,
                dry_run=False,
                sync_tags=True,
                sync_albums=True,
                include_dynamic=True,
            )
        except Exception as e:
            print(f"⚠️  Immich sync failed: {e}")
            print("   You can run `python immich_sync.py` manually.")
    elif not interrupted:
        print("\n💡 No changes were made to Immich (detection results saved locally).")
        print("   • To preview changes safely:  python detect.py --dry-run")
        print("   • To apply Tags and Albums:   python immich_sync.py (or python detect.py --sync)")

    print()


if __name__ == "__main__":
    main()
