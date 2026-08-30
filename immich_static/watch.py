"""
immich_static.watch — Continuous background monitor for newly uploaded videos in Immich library mount.
"""

import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from immich_static.client import ImmichClient
from immich_static.core import (
    CHECKPOINT_FILENAME,
    ENV,
    LOG_FIELDS,
    SENSITIVITY_PRESETS,
    VIDEO_EXTENSIONS,
    Checkpoint,
    detect_video,
)
from immich_static.sync import sync_immich

_stop_event = False


def _handle_signal(sig, frame):
    global _stop_event
    print("\n🛑 Shutting down watcher daemon gracefully...")
    _stop_event = True


def run_cycle(
    folder: Path,
    ckpt: Checkpoint,
    thresholds: dict,
    workers: int,
    immich_client: Optional[ImmichClient] = None
) -> int:
    """Runs a single incremental scan cycle. Returns count of newly detected files."""
    raw_paths = folder.rglob("*")
    all_videos = [
        p for p in raw_paths
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
        and not any(part.startswith(".") for part in p.relative_to(folder).parts)
        and (len(p.relative_to(folder).parts) == 1 or p.relative_to(folder).parts[0] not in ("static", "dynamic", "review"))
    ]

    new_videos = [
        vp for vp in all_videos
        if not ckpt.is_done(str(vp.resolve())) or ckpt.is_error(str(vp.resolve()))
    ]

    if not new_videos:
        return 0

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🎬 Found {len(new_videos)} new video(s) to process.")

    processed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(detect_video, vp, thresholds): vp for vp in new_videos}
        for future in as_completed(futures):
            if _stop_event:
                break
            vp = futures[future]
            full_path = str(vp.resolve())
            try:
                row = future.result()
            except Exception as e:
                row = {f: "" for f in LOG_FIELDS}
                row["decision"] = f"error: {e}"

            row["filename"] = full_path
            ckpt.record(full_path, row)
            processed += 1
            print(f"   [{row.get('decision', '?').upper()}] {vp.name} (motion: {row.get('global_motion_score', '?')})")

    if immich_client and processed > 0 and not _stop_event:
        print("🔄 Syncing new classifications to Immich tags and albums...")
        try:
            sync_immich(immich_client, ckpt, dry_run=False)
        except Exception as e:
            print(f"⚠️  Immich sync failed: {e}")

    return processed


def run_watch(args: Any):
    """Executes the continuous watcher daemon."""
    global _stop_event
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    folder_str = getattr(args, "folder", None) or os.environ.get("IMMICH_LIBRARY_PATH")
    if not folder_str:
        print("❌ Error: Missing Immich library path.")
        print("   Provide folder argument or set IMMICH_LIBRARY_PATH in .env")
        sys.exit(1)

    folder = Path(folder_str).resolve()
    if not folder.is_dir():
        print(f"❌ Directory not found: {folder}")
        sys.exit(1)

    db_path_str = getattr(args, "db_path", None) or os.environ.get("DB_PATH", "./" + CHECKPOINT_FILENAME)
    ckpt_path = Path(db_path_str).resolve()
    ckpt = Checkpoint(ckpt_path)

    sensitivity = getattr(args, "sensitivity", "medium") or "medium"
    thresholds = SENSITIVITY_PRESETS.get(sensitivity, SENSITIVITY_PRESETS["medium"])
    workers = min(getattr(args, "workers", None) or ENV["max_workers"], ENV["max_workers"])
    interval = int(getattr(args, "interval", 300) or 300)

    api_url = getattr(args, "api_url", None) or os.environ.get("IMMICH_API_URL")
    api_key = getattr(args, "api_key", None) or os.environ.get("IMMICH_API_KEY")

    immich_client = None
    if api_url and api_key:
        immich_client = ImmichClient(api_url, api_key)
        try:
            immich_client.ping()
            print("🌐 Connected to Immich API for automatic album syncing.")
        except Exception as e:
            print(f"⚠️  Could not connect to Immich API: {e}. Running in local-only mode.")
            immich_client = None

    print(f"👀 Watching Immich library at: {folder}")
    print(f"⏱  Scan interval: {interval} seconds | Workers: {workers} | Sensitivity: {sensitivity}")
    print("Press Ctrl+C to stop.\n")

    while not _stop_event:
        try:
            run_cycle(folder, ckpt, thresholds, workers, immich_client)
        except Exception as e:
            print(f"⚠️  Error in watch cycle: {e}")

        for _ in range(interval):
            if _stop_event:
                break
            time.sleep(1)

    print("👋 Watcher stopped cleanly.")
