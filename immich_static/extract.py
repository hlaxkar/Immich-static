"""
immich_static.extract — Sharpness-aware still frame extractor for static videos.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from immich_static.client import ImmichClient
from immich_static.core import (
    CHECKPOINT_FILENAME,
    ENV,
    Checkpoint,
    _hard_stop_event,
    _interrupt_event,
    check_system_dependencies,
    extract_one_frame,
    format_duration,
    make_bar,
    setup_interrupt_handlers,
)


def run_extract(args: Any):
    """Executes frame extraction from classified videos in SQLite checkpoint."""
    script_start_time = time.time()
    setup_interrupt_handlers()
    check_system_dependencies(need_cv=True)

    db_path_str = getattr(args, "db_path", None) or os.environ.get("DB_PATH", "./" + CHECKPOINT_FILENAME)
    db_p = Path(db_path_str).resolve()
    if not db_p.exists():
        print(f"❌ Checkpoint database not found: {db_p}")
        print("   Run `immich-static detect` first to classify your video assets.")
        sys.exit(1)

    ckpt = Checkpoint(db_p)
    rows = ckpt.all_rows()

    output_dir_str = getattr(args, "output_dir", None) or os.environ.get("EXTRACT_OUTPUT_DIR", "./extracted_frames")
    output_dir = Path(output_dir_str).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_decision = getattr(args, "target_decision", "static") or "static"
    fmt = getattr(args, "format", None) or os.environ.get("EXTRACT_FORMAT", "jpg")
    quality = int(getattr(args, "quality", None) or os.environ.get("EXTRACT_QUALITY", 95))

    targets = []
    for r in rows:
        dec = r.get("decision", "")
        if target_decision == "all" or dec == target_decision:
            vp = Path(r["filename"])
            if vp.exists():
                orig_name = r.get("original_file_name") or vp.name
                stem = Path(orig_name).stem
                out_path = output_dir / f"{stem}.{fmt}"
                targets.append((r, vp, out_path, orig_name))

    if not targets:
        print(f"ℹ️  No video files with decision='{target_decision}' found in {db_p.name}.")
        sys.exit(0)

    requested = getattr(args, "workers", None) or ENV["max_workers"]
    workers = min(requested, ENV["max_workers"])
    if getattr(args, "workers", None) is not None and args.workers > workers:
        print(f"⚠️  --workers {args.workers} exceeds platform cap; using {workers}")

    skip_existing = getattr(args, "skip_existing", True)
    fresh = getattr(args, "fresh", False)

    def _needs_extraction(t) -> bool:
        r, vp, out_p, orig_name = t
        if not out_p.exists():
            return True
        # If output file exists, re-extract if it has a missing or epoch-zero (1970) EXIF date
        try:
            from PIL import Image
            with Image.open(out_p) as im:
                exif = im.getexif()
                dto = exif.get_ifd(0x8769).get(36867) or exif.get(306)
                if not dto or str(dto).startswith("1970") or str(dto).startswith("1904"):
                    return True
        except Exception:
            return True
        return False

    if skip_existing and not fresh:
        to_process = [t for t in targets if _needs_extraction(t)]
        skipped_count = len(targets) - len(to_process)
    else:
        to_process = targets
        skipped_count = 0

    upload = getattr(args, "upload", False)
    tag_name = os.environ.get("TAG_EXTRACTED", "video:extracted")

    print(f"🎬 Target Group      : {target_decision.upper()} ({len(targets)} candidates)")
    if skipped_count:
        print(f"   ⏭  Already extracted: {skipped_count}")
    print(f"   🔍 To extract       : {len(to_process)}")
    print(f"   📁 Output directory : {output_dir}")
    print(f"   🖼  Format           : {fmt.upper()}"
          + (f" (quality={quality})" if fmt == "jpg" else " (lossless)"))
    print(f"   🧵 Workers          : {workers}")
    print(f"   ☁️  Upload to Immich : {'Yes (Tag: ' + tag_name + ')' if upload else 'No'}")
    print()

    results: List[Dict[str, Any]] = []
    interrupted = False

    if to_process:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for (r, vp, out_path, orig_name) in to_process:
                if _interrupt_event.is_set():
                    break
                tags_to_embed = [tag_name, "static-video", "extracted-frame"]
                date_hint = r.get("date_time_original") or r.get("file_created_at")
                metadata_hint = {
                    "latitude": r.get("latitude"),
                    "longitude": r.get("longitude"),
                    "altitude": r.get("altitude"),
                    "make": r.get("camera_make"),
                    "model": r.get("camera_model"),
                }
                futures[executor.submit(
                    extract_one_frame,
                    video_path=vp,
                    out_path=out_path,
                    fmt=fmt,
                    quality=quality,
                    num_candidates=12,
                    tags=tags_to_embed,
                    original_file_name=orig_name,
                    date_hint=date_hint,
                    metadata_hint=metadata_hint,
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
                        "file": orig_name[:22],
                        "status": result["status"],
                    })

                    if _interrupt_event.is_set():
                        interrupted = True
                        break

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"].startswith("error")]

    print(f"\n── Extraction Summary ──────────────────────")
    print(f"  ✅ Extracted  : {len(ok)}")
    print(f"  ⏭  Skipped   : {len(skipped) + skipped_count}")
    print(f"  ❌ Errors     : {len(errors)}")

    script_end_time = time.time()
    total_time_taken = script_end_time - script_start_time

    print(f"  {'─'*32}")
    print(f"  Total time    : {format_duration(total_time_taken)}")

    if errors:
        print("\n  Failed files:")
        for e in errors[:15]:
            print(f"    • {e['file']} — {e['status']}")

    if interrupted:
        print("\n⚠️  Interrupted — re-run to resume extraction.")
    else:
        print(f"\n🖼  Frames successfully saved to: {output_dir}")

    # Optional Upload to Immich with 'video:extracted' Tag
    if upload and not interrupted:
        api_url = getattr(args, "api_url", None) or os.environ.get("IMMICH_API_URL")
        api_key = getattr(args, "api_key", None) or os.environ.get("IMMICH_API_KEY")
        if not api_url or not api_key:
            print("\n⚠️  --upload requested, but IMMICH_API_URL or IMMICH_API_KEY not configured.")
        else:
            from immich_static.sync import upload_extracted_frames
            client = ImmichClient(api_url, api_key)
            try:
                upload_extracted_frames(client, output_dir, dry_run=False)
            except Exception as e:
                print(f"⚠️  Immich upload/tagging failed: {e}")

    print()
