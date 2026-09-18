"""
immich_static.cli — Unified CLI entrypoint and subcommand dispatcher for immich-static.
"""

import argparse
import os
import sys

from immich_static import __version__
from immich_static.core import CHECKPOINT_FILENAME, ENV, load_dotenv
from immich_static.detect import run_detect
from immich_static.extract import run_extract
from immich_static.server import run_serve
from immich_static.sync import run_pull, run_restore, run_stats, run_sync
from immich_static.test_suite import run_test_cmd
from immich_static.watch import run_watch


def build_parser() -> argparse.ArgumentParser:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="immich-static",
        description="API-First, Zero-Download Static Video Detection and Frame Extraction Toolkit for Immich.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="Subcommands",
        metavar="<command>",
        help="Available commands (run '<command> --help' for details)",
    )

    # ─────────────────────────────────────────────
    # SUBCOMMAND: DETECT / SCAN
    # ─────────────────────────────────────────────
    p_detect = subparsers.add_parser(
        "detect",
        aliases=["scan"],
        help="Scan Immich library mount and classify videos into static, review, and dynamic.",
        description="Scans video files on local disk mount, computes motion metrics, and stores classifications into SQLite.",
    )
    p_detect.add_argument(
        "folder", nargs="?", default=os.environ.get("IMMICH_LIBRARY_PATH"),
        help="Path to Immich library mount on host (or set IMMICH_LIBRARY_PATH in .env)",
    )
    p_detect.add_argument(
        "--library", dest="library_path", default=None,
        help="Alias for folder path",
    )
    p_detect.add_argument(
        "--api-url", default=os.environ.get("IMMICH_API_URL"),
        help="Immich API URL (or set IMMICH_API_URL in .env)",
    )
    p_detect.add_argument(
        "--api-key", default=os.environ.get("IMMICH_API_KEY"),
        help="Immich API Key (or set IMMICH_API_KEY in .env)",
    )
    p_detect.add_argument(
        "--sensitivity", choices=["low", "medium", "high"],
        default=os.environ.get("SENSITIVITY", "medium"),
        help="Sensitivity preset (default: medium)",
    )
    p_detect.add_argument(
        "--workers", type=int,
        default=int(os.environ["WORKERS"]) if os.environ.get("WORKERS") else None,
        help=f"Parallel workers (default: {ENV['max_workers']})",
    )
    p_detect.add_argument(
        "--db", "--db-path", dest="db_path",
        default=os.environ.get("DB_PATH", "./" + CHECKPOINT_FILENAME),
        help="Path to SQLite checkpoint file (default: ./detection_checkpoint.sqlite)",
    )
    p_detect.add_argument(
        "--sync", action="store_true",
        help="Explicitly sync classifications to Immich Tags & Albums after detection",
    )
    p_detect.add_argument(
        "--dry-run", action="store_true",
        help="Preview Immich tag and album changes after scanning without modifying Immich",
    )
    p_detect.add_argument(
        "--export-csv", dest="export_csv", nargs="?", const="detection_log.csv", default=None,
        help="Optional: Export detection results to CSV (default: detection_log.csv)",
    )
    p_detect.add_argument(
        "--fresh", action="store_true",
        help="Ignore checkpoint, re-detect everything",
    )
    p_detect.add_argument(
        "--report", action="store_true",
        help="Print detailed confidence score breakdown",
    )
    p_detect.add_argument(
        "--debug", action="store_true",
        help="Print ffmpeg stderr for any video that fails frame extraction",
    )
    p_detect.add_argument(
        "--prune", action="store_true",
        help="Remove orphaned records from SQLite checkpoint for videos no longer in Immich",
    )
    p_detect.set_defaults(func=run_detect)

    # ─────────────────────────────────────────────
    # SUBCOMMAND: SYNC
    # ─────────────────────────────────────────────
    p_sync = subparsers.add_parser(
        "sync",
        help="Sync classified videos from SQLite checkpoint to Immich Tags & Albums.",
        description="Applies tags (video:static, video:review, video:dynamic) and organizes videos into albums.",
    )
    p_sync.add_argument(
        "--api-url", default=os.environ.get("IMMICH_API_URL"),
        help="Immich API URL (or set IMMICH_API_URL in .env)",
    )
    p_sync.add_argument(
        "--api-key", default=os.environ.get("IMMICH_API_KEY"),
        help="Immich API Key (or set IMMICH_API_KEY in .env)",
    )
    p_sync.add_argument(
        "--db", "--db-path", dest="db_path",
        default=os.environ.get("DB_PATH", "./" + CHECKPOINT_FILENAME),
        help="Path to SQLite checkpoint database (default: ./detection_checkpoint.sqlite)",
    )
    p_sync.add_argument(
        "--no-tags", action="store_true",
        help="Skip applying tags to assets",
    )
    p_sync.add_argument(
        "--no-albums", action="store_true",
        help="Skip adding assets to albums",
    )
    p_sync.add_argument(
        "--include-dynamic", action=argparse.BooleanOptionalAction, default=True,
        help="Also tag and create album for dynamic videos (default: True)",
    )
    p_sync.add_argument(
        "--upload-extracted", action="store_true",
        help="Also upload local extracted frames in EXTRACT_OUTPUT_DIR to Immich",
    )
    p_sync.add_argument(
        "--prune", action="store_true",
        help="Remove orphaned records from SQLite checkpoint for videos no longer in Immich",
    )
    p_sync.add_argument(
        "--dry-run", action="store_true",
        help="Preview operations without modifying Immich",
    )
    p_sync.set_defaults(func=run_sync)

    # ─────────────────────────────────────────────
    # SUBCOMMAND: STATS / STORAGE
    # ─────────────────────────────────────────────
    p_stats = subparsers.add_parser(
        "stats",
        aliases=["storage", "info"],
        help="Display video counts, duration, and storage space metrics.",
        description="Queries Immich API / SQLite checkpoint to calculate album sizes and estimated storage savings.",
    )
    p_stats.add_argument(
        "--api-url", default=os.environ.get("IMMICH_API_URL"),
        help="Immich API URL (or set IMMICH_API_URL in .env)",
    )
    p_stats.add_argument(
        "--api-key", default=os.environ.get("IMMICH_API_KEY"),
        help="Immich API Key (or set IMMICH_API_KEY in .env)",
    )
    p_stats.add_argument(
        "--db", "--db-path", dest="db_path",
        default=os.environ.get("DB_PATH", "./" + CHECKPOINT_FILENAME),
        help="Path to SQLite checkpoint database",
    )
    p_stats.set_defaults(func=run_stats)

    # ─────────────────────────────────────────────
    # SUBCOMMAND: PULL / PULL-ALBUMS
    # ─────────────────────────────────────────────
    p_pull = subparsers.add_parser(
        "pull",
        aliases=["pull-albums"],
        help="Pull Immich album modifications back into local SQLite checkpoint.",
        description="Syncs manual album reclassifications from Immich UI back to the local SQLite database.",
    )
    p_pull.add_argument(
        "--api-url", default=os.environ.get("IMMICH_API_URL"),
        help="Immich API URL (or set IMMICH_API_URL in .env)",
    )
    p_pull.add_argument(
        "--api-key", default=os.environ.get("IMMICH_API_KEY"),
        help="Immich API Key (or set IMMICH_API_KEY in .env)",
    )
    p_pull.add_argument(
        "--db", "--db-path", dest="db_path",
        default=os.environ.get("DB_PATH", "./" + CHECKPOINT_FILENAME),
        help="Path to SQLite checkpoint database",
    )
    p_pull.add_argument(
        "--no-tag-sync-on-pull", action="store_true",
        help="Skip updating Immich tags when pulling album changes",
    )
    p_pull.add_argument(
        "--dry-run", action="store_true",
        help="Preview pull changes without modifying SQLite checkpoint",
    )
    p_pull.set_defaults(func=run_pull)

    # ─────────────────────────────────────────────
    # SUBCOMMAND: EXTRACT
    # ─────────────────────────────────────────────
    p_extract = subparsers.add_parser(
        "extract",
        help="Extract the highest sharpness frame from classified static videos.",
        description="Uses Laplacian variance to select and export the sharpest still frame with EXIF timestamps preserved.",
    )
    p_extract.add_argument(
        "--db", "--db-path", dest="db_path",
        default=os.environ.get("DB_PATH", "./" + CHECKPOINT_FILENAME),
        help="Path to SQLite checkpoint database (default: ./detection_checkpoint.sqlite)",
    )
    p_extract.add_argument(
        "--target-decision", choices=["static", "review", "all"], default="static",
        help="Which classification group to extract (default: static)",
    )
    p_extract.add_argument(
        "--output-dir",
        default=os.environ.get("EXTRACT_OUTPUT_DIR", "./extracted_frames"),
        help="Directory to save extracted frames (default: ./extracted_frames/)",
    )
    p_extract.add_argument(
        "--format", choices=["png", "jpg"],
        default=os.environ.get("EXTRACT_FORMAT", "jpg"),
        help="Output format (default: jpg)",
    )
    p_extract.add_argument(
        "--quality", type=int,
        default=int(os.environ.get("EXTRACT_QUALITY", 95)),
        help="JPG quality 1-100 (default: 95)",
    )
    p_extract.add_argument(
        "--workers", type=int,
        default=int(os.environ["WORKERS"]) if os.environ.get("WORKERS") else None,
        help=f"Parallel workers (default: {ENV['max_workers']})",
    )
    p_extract.add_argument(
        "--upload", action="store_true",
        help="Upload newly extracted frames to Immich and apply '[Extracted] Photos' album & tag",
    )
    p_extract.add_argument(
        "--api-url", default=os.environ.get("IMMICH_API_URL"),
        help="Immich API URL (required if --upload is used)",
    )
    p_extract.add_argument(
        "--api-key", default=os.environ.get("IMMICH_API_KEY"),
        help="Immich API Key (required if --upload is used)",
    )
    p_extract.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True,
        help="Skip videos whose frame already exists in output dir (default: True)",
    )
    p_extract.add_argument(
        "--fresh", action="store_true",
        help="Re-extract everything, ignoring existing files",
    )
    p_extract.set_defaults(func=run_extract)

    # ─────────────────────────────────────────────
    # SUBCOMMAND: RESTORE / RECALCULATE
    # ─────────────────────────────────────────────
    p_restore = subparsers.add_parser(
        "restore",
        aliases=["recalculate"],
        help="Recalculate classifications from stored motion scores in SQLite.",
        description="Re-applies sensitivity threshold logic against previously analyzed motion data without re-decoding videos.",
    )
    p_restore.add_argument(
        "--db", "--db-path", dest="db_path",
        default=os.environ.get("DB_PATH", "./" + CHECKPOINT_FILENAME),
        help="Path to SQLite checkpoint database",
    )
    p_restore.add_argument(
        "--sensitivity", choices=["low", "medium", "high"],
        default=os.environ.get("SENSITIVITY", "medium"),
        help="Sensitivity preset (default: medium)",
    )
    p_restore.set_defaults(func=run_restore)

    # ─────────────────────────────────────────────
    # SUBCOMMAND: SERVE / WEBHOOK
    # ─────────────────────────────────────────────
    p_serve = subparsers.add_parser(
        "serve",
        aliases=["webhook"],
        help="Run real-time webhook daemon for Immich workflow notifications.",
        description="Listens for Immich Workflows AssetCreate webhooks, processes newly uploaded videos on the fly, and syncs tags & albums.",
    )
    p_serve.add_argument(
        "folder", nargs="?", default=os.environ.get("IMMICH_LIBRARY_PATH"),
        help="Path to mounted Immich library directory (optional; falls back to API streaming)",
    )
    p_serve.add_argument(
        "--host", default=os.environ.get("WEBHOOK_HOST", "0.0.0.0"),
        help="Host interface to bind HTTP webhook server (default: 0.0.0.0)",
    )
    p_serve.add_argument(
        "--port", type=int, default=int(os.environ.get("WEBHOOK_PORT", 8080)),
        help="Port to listen for webhooks (default: 8080)",
    )
    p_serve.add_argument(
        "--secret", default=os.environ.get("WEBHOOK_SECRET"),
        help="Shared secret token for webhook authentication (header X-Webhook-Secret)",
    )
    p_serve.add_argument(
        "--remote-prefix", default=os.environ.get("IMMICH_REMOTE_PATH_PREFIX"),
        help="Remote path prefix to remap (e.g. Docker container path '/usr/src/app/upload')",
    )
    p_serve.add_argument(
        "--local-prefix", default=os.environ.get("IMMICH_LOCAL_PATH_PREFIX"),
        help="Local path prefix to replace remote prefix with (e.g. '/mnt/storage/immich')",
    )
    p_serve.add_argument(
        "--sensitivity", choices=["low", "medium", "high"],
        default=os.environ.get("SENSITIVITY", "medium"),
        help="Detection sensitivity preset (default: medium)",
    )
    p_serve.add_argument(
        "--workers", type=int, default=2,
        help="Concurrent background video processing workers (default: 2)",
    )
    p_serve.add_argument(
        "--include-dynamic", action=argparse.BooleanOptionalAction, default=True,
        help="Also tag and create album for dynamic videos (default: True)",
    )
    p_serve.add_argument(
        "--extract", action="store_true",
        help="Extract sharpest still frame for static videos and upload to Immich",
    )
    p_serve.add_argument(
        "--extract-dir", default=os.environ.get("EXTRACT_OUTPUT_DIR", "./extracted_frames"),
        help="Directory to save extracted still frames",
    )
    p_serve.add_argument(
        "--db", "--db-path", dest="db_path",
        default=os.environ.get("DB_PATH", "./" + CHECKPOINT_FILENAME),
        help="Path to SQLite checkpoint database",
    )
    p_serve.add_argument(
        "--api-url", default=os.environ.get("IMMICH_API_URL"),
        help="Immich API URL (required)",
    )
    p_serve.add_argument(
        "--api-key", default=os.environ.get("IMMICH_API_KEY"),
        help="Immich API Key (required)",
    )
    p_serve.add_argument(
        "--dry-run", action="store_true",
        help="Process videos and log decisions without modifying Immich tags/albums",
    )
    p_serve.set_defaults(func=run_serve)

    # ─────────────────────────────────────────────
    # SUBCOMMAND: WATCH
    # ─────────────────────────────────────────────
    p_watch = subparsers.add_parser(
        "watch",
        help="Continuously monitor Immich library for new video uploads in background.",
        description="Runs an incremental polling loop to classify newly added video files.",
    )
    p_watch.add_argument(
        "folder", nargs="?", default=os.environ.get("IMMICH_LIBRARY_PATH"),
        help="Path to mounted Immich library directory",
    )
    p_watch.add_argument(
        "--interval", type=int, default=300,
        help="Polling interval in seconds (default: 300)",
    )
    p_watch.add_argument(
        "--sensitivity", choices=["low", "medium", "high"], default="medium",
    )
    p_watch.add_argument(
        "--workers", type=int, default=None,
        help=f"Parallel workers (default: {ENV['max_workers']})",
    )
    p_watch.add_argument(
        "--db", "--db-path", dest="db_path", default=None,
        help="Path to SQLite checkpoint database",
    )
    p_watch.add_argument(
        "--api-url", default=os.environ.get("IMMICH_API_URL"),
        help="Immich API URL for automatic syncing",
    )
    p_watch.add_argument(
        "--api-key", default=os.environ.get("IMMICH_API_KEY"),
        help="Immich API Key for automatic syncing",
    )
    p_watch.set_defaults(func=run_watch)

    # ─────────────────────────────────────────────
    # SUBCOMMAND: TEST
    # ─────────────────────────────────────────────
    p_test = subparsers.add_parser(
        "test",
        help="Run automated synthetic test suite.",
        description="Generates synthetic static and dynamic test videos and verifies metadata, detection, checkpointing, and extraction.",
    )
    p_test.set_defaults(func=run_test_cmd)

    return parser


def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(0)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
