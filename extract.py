#!/usr/bin/env python3
"""
extract.py — Extract the highest sharpness frame from static videos in Immich.

Pulls classified static videos from the SQLite checkpoint and extracts the
single sharpest frame using Laplacian variance. Optionally uploads and tags
the extracted images in Immich.

Usage:
    # Run with defaults (reads DB and settings from .env):
    python extract.py

    # Extract and upload directly to Immich with "Extracted" tag:
    python extract.py --upload

    # Custom options:
    python extract.py --target-decision review --output-dir ./extracted_review_frames
    python extract.py --format png

Options:
    --db PATH               Path to SQLite checkpoint (default: from .env or ./detection_checkpoint.sqlite)
    --target-decision       static | review | all  (default: static)
    --output-dir PATH       Where to save extracted frames (default: ./extracted_frames/)
    --format png|jpg        Output format (default: jpg)
    --quality 1-100         JPG quality (default: 95)
    --workers N             Parallel workers (default: auto)
    --upload                Upload extracted frames to Immich and apply 'Extracted' tag
    --skip-existing         Skip videos whose frame already exists (default: True)
    --fresh                 Re-extract everything, ignoring skip list
"""

import argparse
import os
import shutil
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

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
    ENV, VIDEO_EXTENSIONS, CHECKPOINT_FILENAME, Checkpoint, extract_one_frame,
)
from immich_sync import ImmichClient

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
                "\n\n⚠️  Interrupt received — finishing current extractions then stopping...\n"
                "   Ctrl+C again to force quit immediately.\n"
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
# MAIN
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract the highest sharpness frame from static videos in Immich.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--db", "--db-path", dest="db_path",
                   default=os.environ.get("DB_PATH", "./" + CHECKPOINT_FILENAME),
                   help="Path to SQLite checkpoint database (default: ./detection_checkpoint.sqlite)")
    p.add_argument("--target-decision", choices=["static", "review", "all"], default="static",
                   help="Which classification group to extract (default: static)")
    p.add_argument("--output-dir",
                   default=os.environ.get("EXTRACT_OUTPUT_DIR", "./extracted_frames"),
                   help="Directory to save extracted frames (default: ./extracted_frames/)")
    p.add_argument("--format", choices=["png", "jpg"],
                   default=os.environ.get("EXTRACT_FORMAT", "jpg"),
                   help="Output format (default: jpg)")
    p.add_argument("--quality", type=int,
                   default=int(os.environ.get("EXTRACT_QUALITY", 95)),
                   help="JPG quality 1-100 (default: 95)",
                   choices=range(1, 101), metavar="1-100")
    p.add_argument("--workers", type=int,
                   default=int(os.environ["WORKERS"]) if os.environ.get("WORKERS") else None,
                   help=f"Parallel workers (default: {ENV['max_workers']})")
    p.add_argument("--upload", action="store_true",
                   help="Upload newly extracted frames to Immich and apply 'Extracted' tag")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip videos whose frame already exists in output dir (default: True)")
    p.add_argument("--fresh", action="store_true",
                   help="Re-extract everything, ignoring skip list")
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

    db_p = Path(args.db_path).resolve()
    if not db_p.exists():
        print(f"❌ Checkpoint database not found: {db_p}")
        print("   Run `python detect.py` first to classify your video assets.")
        sys.exit(1)

    ckpt = Checkpoint(db_p)
    rows = ckpt.all_rows()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    for r in rows:
        dec = r.get("decision", "")
        if args.target_decision == "all" or dec == args.target_decision:
            vp = Path(r["filename"])
            if vp.exists():
                orig_name = r.get("original_file_name") or vp.name
                stem = Path(orig_name).stem
                out_path = output_dir / f"{stem}.{args.format}"
                targets.append((r, vp, out_path, orig_name))

    if not targets:
        print(f"No video files with decision='{args.target_decision}' found in {db_p.name}.")
        sys.exit(0)

    requested = args.workers or ENV["max_workers"]
    workers   = min(requested, ENV["max_workers"])
    if args.workers is not None and args.workers > workers:
        print(f"⚠️  --workers {args.workers} exceeds platform cap; using {workers}")

    if args.skip_existing and not args.fresh:
        to_process = [t for t in targets if not t[2].exists()]
        skipped_count = len(targets) - len(to_process)
    else:
        to_process    = targets
        skipped_count = 0

    tag_name = os.environ.get("TAG_EXTRACTED", "video:extracted")
    print(f"🎬 Target Group      : {args.target_decision.upper()} ({len(targets)} candidates)")
    if skipped_count:
        print(f"   ⏭  Already extracted: {skipped_count}")
    print(f"   🔍 To extract       : {len(to_process)}")
    print(f"   📁 Output directory : {output_dir}")
    print(f"   🖼  Format           : {args.format.upper()}"
          + (f" (quality={args.quality})" if args.format == "jpg" else " (lossless)"))
    print(f"   🧵 Workers          : {workers}")
    print(f"   ☁️  Upload to Immich : {'Yes (Tag: ' + tag_name + ')' if args.upload else 'No'}")
    print()

    results     = []
    interrupted = False

    if to_process:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for (r, vp, out_path, orig_name) in to_process:
                if _interrupt_event.is_set():
                    break
                futures[executor.submit(
                    extract_one_frame, vp, out_path,
                    args.format, args.quality
                )] = (vp, out_path, orig_name)

            with make_bar(len(futures), "Extracting") as pbar:
                for future in as_completed(futures):
                    if _hard_stop_event.is_set():
                        break

                    (vp, out_path, orig_name) = futures[future]
                    try:
                        result = future.result()
                        result["file"] = orig_name
                    except Exception as e:
                        result = {"file": orig_name, "status": f"error: {e}", "output": str(out_path)}

                    results.append(result)
                    pbar.update(1)
                    pbar.set_postfix({
                        "file":   orig_name[:22],
                        "status": result["status"],
                    })

                    if _interrupt_event.is_set():
                        interrupted = True
                        break

    ok      = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors  = [r for r in results if r["status"].startswith("error")]

    print(f"\n── Extraction Summary ──────────────────────")
    print(f"  ✅ Extracted  : {len(ok)}")
    print(f"  ⏭  Skipped   : {len(skipped) + skipped_count}")
    print(f"  ❌ Errors     : {len(errors)}")

    def fmt_time(seconds: float) -> str:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0: return f"{int(h)}h {int(m)}m {s:.1f}s"
        if m > 0: return f"{int(m)}m {s:.1f}s"
        return f"{s:.2f}s"

    script_end_time = time.time()
    total_time_taken = script_end_time - script_start_time

    print(f"  {'─'*32}")
    print(f"  Total time    : {fmt_time(total_time_taken)}")

    if errors:
        print("\n  Failed files:")
        for e in errors[:15]:
            print(f"    • {e['file']} — {e['status']}")

    if interrupted:
        print("\n⚠️  Interrupted — re-run to resume extraction.")
    else:
        print(f"\n🖼  Frames successfully saved to: {output_dir}")

    # Optional Upload to Immich with 'video:extracted' Tag
    if args.upload and not interrupted:
        api_url = os.environ.get("IMMICH_API_URL")
        api_key = os.environ.get("IMMICH_API_KEY")
        if not api_url or not api_key:
            print("\n⚠️  --upload requested, but IMMICH_API_URL or IMMICH_API_KEY not configured in .env.")
        else:
            # Find all extracted image files in output_dir
            images_to_upload = sorted([
                p for p in output_dir.iterdir()
                if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
            ])
            if not images_to_upload:
                print(f"\n⚠️  No extracted images found in {output_dir} to upload.")
            else:
                print(f"\n☁️  Uploading {len(images_to_upload)} extracted frames to Immich...")
                client = ImmichClient(api_url, api_key)
                try:
                    target_tag_name = os.environ.get("TAG_EXTRACTED", "video:extracted")
                    tags = client.get_tags()
                    extracted_tag = next((t for t in tags if t.get("name") == target_tag_name), None)
                    if not extracted_tag:
                        extracted_tag = client.create_tag(target_tag_name)
                    tag_id = extracted_tag["id"]

                    uploaded_ids = []
                    for img_path in images_to_upload:
                        res = client.upload_asset(img_path, created_at=img_path.stat().st_mtime)
                        if res and "id" in res:
                            aid = res["id"]
                            status_str = res.get("status", "created")
                            uploaded_ids.append(aid)
                            print(f"   Uploaded: {img_path.name} -> Immich ID: {aid} ({status_str})")
                        else:
                            print(f"   Uploaded: {img_path.name} -> (Response: {res})")

                    if uploaded_ids:
                        print(f"🏷  Applying '{target_tag_name}' tag to {len(uploaded_ids)} uploaded images...")
                        client.tag_assets(tag_id, uploaded_ids)
                        print(f"✅ Successfully tagged with '{target_tag_name}'!")

                        # Also add to '[Extracted] Photos' album for instant visibility in Immich UI
                        album_name = "[Extracted] Photos"
                        albums = client.get_albums()
                        target_album = next((a for a in albums if a.get("albumName") == album_name), None)
                        if not target_album:
                            target_album = client.create_album(album_name)
                        album_id = target_album["id"]
                        client.add_assets_to_album(album_id, uploaded_ids)
                        print(f"📁 Added {len(uploaded_ids)} images to Album '{album_name}'")

                        # Ensure all assets in the extracted album have the tag applied
                        album_asset_ids = client.get_album_assets(album_id)
                        all_to_tag = list(set(uploaded_ids) | album_asset_ids)
                        client.tag_assets(tag_id, all_to_tag)
                        print(f"🏷  Verified tag '{target_tag_name}' across all {len(all_to_tag)} assets in '{album_name}'")
                except Exception as e:
                    print(f"⚠️  Immich upload/tagging failed: {e}")

    print()


if __name__ == "__main__":
    main()
