#!/usr/bin/env python3
"""
watch.py — Continuous background monitor for newly uploaded videos in Immich library mount.

Features:
- Continuously scans the mounted Immich library at a configurable interval.
- Detects and classifies only new, unprocessed video uploads.
- Automatically syncs new static/review videos to Immich albums if API credentials are provided.
- Low-overhead, resumable, and container-ready.

Usage:
    python watch.py /path/to/immich/library --interval 300 [options]

Options:
    --interval SECONDS   Seconds between scan cycles (default: 300)
    --sensitivity LEVEL  low | medium | high (default: medium)
    --workers N          Parallel workers for detection
    --db PATH            Path to SQLite checkpoint db (default: ./detection_checkpoint.sqlite)
    --api-url URL        Immich URL for auto-syncing albums (optional)
    --api-key KEY        Immich API Key for auto-syncing albums (optional)
"""

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from core import (
    ENV, SENSITIVITY_PRESETS, VIDEO_EXTENSIONS,
    CHECKPOINT_FILENAME, LOG_FIELDS,
    Checkpoint, detect_video,
)
from immich_sync import ImmichClient, sync_immich

_stop_event = False

def _handle_signal(sig, frame):
    global _stop_event
    print("\n🛑 Shutting down watcher daemon gracefully...")
    _stop_event = True

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


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

    from concurrent.futures import ThreadPoolExecutor, as_completed

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


def parse_args():
    p = argparse.ArgumentParser(
        description="Continuous background watcher for Immich library mount.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("folder", nargs="?", default=os.environ.get("IMMICH_LIBRARY_PATH"),
                   help="Path to mounted Immich library directory (or set IMMICH_LIBRARY_PATH in .env)")
    p.add_argument("--interval", type=int, default=300,
                   help="Polling interval in seconds (default: 300)")
    p.add_argument("--sensitivity", choices=["low", "medium", "high"], default="medium")
    p.add_argument("--workers", type=int, default=None,
                   help=f"Parallel workers (default: {ENV['max_workers']})")
    p.add_argument("--db", "--db-path", dest="db_path", default=None,
                   help="Path to SQLite checkpoint database")
    p.add_argument("--api-url", default=os.environ.get("IMMICH_API_URL"),
                   help="Immich API URL (or set IMMICH_API_URL in .env)")
    p.add_argument("--api-key", default=os.environ.get("IMMICH_API_KEY"),
                   help="Immich API Key (or set IMMICH_API_KEY in .env)")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.folder:
        print("❌ Error: Missing Immich library path.")
        print("   Provide folder argument or set IMMICH_LIBRARY_PATH in .env")
        sys.exit(1)

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"❌ Directory not found: {folder}")
        sys.exit(1)

    ckpt_path = Path(args.db_path).resolve() if args.db_path else Path("./" + CHECKPOINT_FILENAME).resolve()
    ckpt = Checkpoint(ckpt_path)
    thresholds = SENSITIVITY_PRESETS[args.sensitivity]
    workers = min(args.workers or ENV["max_workers"], ENV["max_workers"])

    immich_client = None
    if args.api_url and args.api_key:
        immich_client = ImmichClient(args.api_url, args.api_key)
        try:
            immich_client.ping()
            print("🌐 Connected to Immich API for automatic album syncing.")
        except Exception as e:
            print(f"⚠️  Could not connect to Immich API: {e}. Running in local-only mode.")
            immich_client = None

    print(f"👀 Watching Immich library at: {folder}")
    print(f"⏱  Scan interval: {args.interval} seconds | Workers: {workers} | Sensitivity: {args.sensitivity}")
    print("Press Ctrl+C to stop.\n")

    while not _stop_event:
        try:
            run_cycle(folder, ckpt, thresholds, workers, immich_client)
        except Exception as e:
            print(f"⚠️  Error in watch cycle: {e}")

        # Sleep in small slices to respond promptly to stop signal
        for _ in range(args.interval):
            if _stop_event:
                break
            time.sleep(1)

    print("👋 Watcher stopped cleanly.")


if __name__ == "__main__":
    main()
