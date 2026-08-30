"""
immich_static.detect — Multi-threaded Immich library scanner and static video classifier.
"""

import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from immich_static.client import ImmichClient
from immich_static.core import (
    CHECKPOINT_FILENAME,
    ENV,
    LOG_FIELDS,
    SENSITIVITY_PRESETS,
    Checkpoint,
    _hard_stop_event,
    _interrupt_event,
    check_system_dependencies,
    detect_video,
    format_duration,
    make_bar,
    resolve_local_video_path,
    setup_interrupt_handlers,
)
from immich_static.sync import sync_immich


def print_report(rows: List[Dict[str, Any]]):
    """Prints a detailed classification report grouped by category."""
    decisions = ["static", "review", "dynamic"]
    grouped = {d: [] for d in decisions}
    errors = []

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


def print_summary(rows: List[Dict[str, Any]], interrupted: bool, total_time: float = 0.0):
    """Prints high-level classification summary counts."""
    counts: Dict[str, int] = {}
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
    print(f"  Total Scan Time                           : {format_duration(total_time):>5}")
    print(f"  {'─'*50}")

    if interrupted:
        print("\n⚠️  Interrupted — checkpoint saved. Re-run anytime to resume.")
    print()


def run_detect(args: Any):
    """Main execution function for detection and library classification."""
    script_start_time = time.time()
    setup_interrupt_handlers()
    check_system_dependencies(need_cv=True)

    api_url = getattr(args, "api_url", None) or os.environ.get("IMMICH_API_URL")
    api_key = getattr(args, "api_key", None) or os.environ.get("IMMICH_API_KEY")
    library_dir_str = getattr(args, "library_path", None) or getattr(args, "folder", None) or os.environ.get("IMMICH_LIBRARY_PATH")

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

    sensitivity = getattr(args, "sensitivity", "medium") or "medium"
    thresholds = SENSITIVITY_PRESETS[sensitivity]
    requested = getattr(args, "workers", None) or ENV["max_workers"]
    workers = min(requested, ENV["max_workers"])
    db_path_str = getattr(args, "db_path", None) or os.environ.get("DB_PATH", "./" + CHECKPOINT_FILENAME)
    ckpt_path = Path(db_path_str).resolve()
    export_csv = getattr(args, "export_csv", None)
    csv_path = Path(export_csv).resolve() if export_csv else None

    ckpt = Checkpoint(ckpt_path)
    if getattr(args, "fresh", False):
        ckpt.clear()
        print("🔄 Checkpoint cleared — starting fresh.\n")

    ckpt.save_meta(
        sensitivity=sensitivity,
        started_at=datetime.now().isoformat(),
        library_dir=str(library_dir),
        api_url=api_url,
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
    print(f"   • Sensitivity Profile : {sensitivity}")
    print(f"   • Parallel Workers    : {workers}")
    print(f"   • SQLite Checkpoint   : {ckpt_path}")
    print()

    interrupted = False
    debug = getattr(args, "debug", False)

    # 5. Multi-Threaded Processing on Local Disk
    if to_detect:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for (asset, vp) in to_detect:
                if _interrupt_event.is_set():
                    break
                futures[executor.submit(detect_video, vp, thresholds, debug)] = (asset, vp)

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
                        "file": vp.name[:20],
                        "result": row.get("decision", "?"),
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

    if getattr(args, "report", False):
        print_report(all_rows)

    print(f"💾 Checkpoint DB : {ckpt_path}")
    if csv_path:
        print(f"📄 Exported CSV  : {csv_path}")

    # 6. Optional Tags & Albums Sync
    dry_run = getattr(args, "dry_run", False)
    sync = getattr(args, "sync", False)

    if dry_run and not interrupted:
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
    elif sync and not interrupted:
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
            print("   You can run `immich-static sync` manually.")
    elif not interrupted:
        print("\n💡 No changes were made to Immich (detection results saved locally).")
        print("   • To preview changes safely:  immich-static sync --dry-run")
        print("   • To apply Tags and Albums:   immich-static sync")

    print()
