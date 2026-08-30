"""
immich_static.sync — Immich REST API bridge for syncing tags, albums, metrics, and frame uploads.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from immich_static.client import ImmichClient
from immich_static.core import (
    CHECKPOINT_FILENAME,
    SENSITIVITY_PRESETS,
    Checkpoint,
    format_bytes,
    format_duration,
)


def sync_immich(
    client: ImmichClient,
    ckpt: Checkpoint,
    dry_run: bool = False,
    sync_tags: bool = True,
    sync_albums: bool = True,
    include_dynamic: bool = False,
):
    """Syncs classified assets from SQLite checkpoint to Immich Tags and Albums."""
    # 1. Fetch Immich Assets & Build Lookup Maps
    immich_videos = client.get_all_video_assets()

    path_to_asset: Dict[str, Dict[str, Any]] = {}
    name_to_assets: Dict[str, List[Dict[str, Any]]] = {}

    for asset in immich_videos:
        aid = asset.get("id")
        orig_path = asset.get("originalPath") or ""
        orig_name = asset.get("originalFileName") or Path(orig_path).name

        if orig_path:
            norm_path = os.path.normpath(orig_path)
            path_to_asset[norm_path] = asset
            path_to_asset[Path(orig_path).name] = asset

        if orig_name:
            name_to_assets.setdefault(orig_name, []).append(asset)

    # 2. Match Checkpoint Entries
    all_rows = ckpt.all_rows()
    print(f"📊 Checkpoint contains {len(all_rows)} classified files.")

    grouped_asset_ids: Dict[str, List[str]] = {
        "static": [],
        "review": [],
        "dynamic": [],
    }

    matched_count = 0
    unmatched_count = 0

    for r in all_rows:
        decision = r.get("decision")
        if decision not in ("static", "review", "dynamic"):
            continue
        if decision == "dynamic" and not include_dynamic:
            continue

        aid = r.get("asset_id")
        if aid and aid in {a["id"] for a in immich_videos}:
            grouped_asset_ids[decision].append(aid)
            matched_count += 1
            continue

        filename_str = r.get("filename", "")
        file_path = Path(filename_str)
        file_name = file_path.name
        matched_asset = None

        norm_full = os.path.normpath(filename_str)
        if norm_full in path_to_asset:
            matched_asset = path_to_asset[norm_full]
        elif file_name in name_to_assets and len(name_to_assets[file_name]) == 1:
            matched_asset = name_to_assets[file_name][0]
        else:
            for p, a in path_to_asset.items():
                if filename_str.endswith(p) or p.endswith(str(file_path.name)):
                    matched_asset = a
                    break

        if matched_asset:
            aid = matched_asset["id"]
            grouped_asset_ids[decision].append(aid)
            matched_count += 1
            if not r.get("asset_id"):
                r["asset_id"] = aid
                ckpt.record(aid, r)
        else:
            unmatched_count += 1

    print(f"🔗 Match Results:")
    print(f"   • Matched Immich Assets : {matched_count}")
    print(f"   • Unmatched Local Files : {unmatched_count}")
    print(f"   • Static candidates     : {len(grouped_asset_ids['static'])}")
    print(f"   • Review candidates     : {len(grouped_asset_ids['review'])}")
    if include_dynamic:
        print(f"   • Dynamic candidates    : {len(grouped_asset_ids['dynamic'])}")
    print()

    tag_name_map = {
        "static": os.environ.get("TAG_STATIC", "video:static"),
        "review": os.environ.get("TAG_REVIEW", "video:review"),
        "dynamic": os.environ.get("TAG_DYNAMIC", "video:dynamic"),
    }

    album_name_map = {
        "static": "[Static] Videos",
        "review": "[Review] Videos",
        "dynamic": "[Dynamic] Videos",
    }

    # 3. Apply Tags to Assets
    if sync_tags:
        print("🏷  Syncing Immich Tags...")
        existing_tags = client.get_tags()
        tags_by_name = {t.get("name"): t for t in existing_tags}

        for decision, asset_ids in grouped_asset_ids.items():
            if not asset_ids:
                continue
            target_tag_name = tag_name_map[decision]

            tag = tags_by_name.get(target_tag_name)
            if not tag:
                if dry_run:
                    print(f"   🔍 [DRY-RUN] Would create Tag '{target_tag_name}'")
                    tag_id = "dry-run-tag-id"
                else:
                    print(f"   🏷  Creating Tag '{target_tag_name}'...")
                    tag = client.create_tag(target_tag_name)
                    tag_id = tag["id"]
            else:
                tag_id = tag["id"]

            if dry_run:
                print(f"   🔍 [DRY-RUN] Would apply tag '{target_tag_name}' to {len(asset_ids)} assets")
            else:
                print(f"   🏷  Applying tag '{target_tag_name}' to {len(asset_ids)} assets...")
                client.tag_assets(tag_id, asset_ids)
                print(f"   ✅ Tagged '{target_tag_name}' ({len(asset_ids)} assets)")
        print()

    # 4. Populate Albums
    if sync_albums:
        print("📁 Syncing Immich Albums...")
        existing_albums = client.get_albums()
        albums_by_name = {a.get("albumName"): a for a in existing_albums}

        for decision, asset_ids in grouped_asset_ids.items():
            if not asset_ids:
                continue
            target_album_name = album_name_map[decision]

            album = albums_by_name.get(target_album_name)
            if not album:
                if dry_run:
                    print(f"   🔍 [DRY-RUN] Would create Album '{target_album_name}'")
                    album_id = "dry-run-album-id"
                    existing_in_album = set()
                else:
                    print(f"   📁 Creating Album '{target_album_name}'...")
                    album = client.create_album(target_album_name)
                    album_id = album["id"]
                    existing_in_album = set()
            else:
                album_id = album["id"]
                existing_in_album = client.get_album_assets(album_id) if not dry_run else set()

            to_add = [aid for aid in asset_ids if aid not in existing_in_album]

            print(f"   📦 Album '{target_album_name}':")
            print(f"      • Total classified : {len(asset_ids)}")
            print(f"      • Already in album : {len(existing_in_album)}")
            print(f"      • New to add       : {len(to_add)}")

            if to_add:
                if dry_run:
                    print(f"      🔍 [DRY-RUN] Would add {len(to_add)} assets to '{target_album_name}'")
                else:
                    print(f"      🚀 Adding {len(to_add)} assets to '{target_album_name}'...")
                    client.add_assets_to_album(album_id, to_add)
                    print(f"      ✅ Done.")
        print()


def restore_checkpoint_decisions(ckpt: Checkpoint, sensitivity: str = "medium") -> int:
    """Restores classifications in SQLite checkpoint database from recorded motion scores."""
    thresholds = SENSITIVITY_PRESETS.get(sensitivity, SENSITIVITY_PRESETS["medium"])
    static_max_m = thresholds["static_max_motion"]
    static_max_z = thresholds["static_max_active_zones"]
    review_max_m = thresholds["review_max_motion"]
    review_max_z = thresholds["review_max_active_zones"]

    rows = ckpt.all_rows()
    restored_counts = {"static": 0, "review": 0, "dynamic": 0}

    for r in rows:
        g_str = r.get("global_motion_score")
        z_str = r.get("active_zone_ratio")
        if not g_str or not z_str:
            continue
        try:
            global_motion = float(g_str)
            active_ratio = float(z_str)
        except (ValueError, TypeError):
            continue

        if global_motion <= static_max_m and active_ratio <= static_max_z:
            new_dec = "static"
        elif global_motion <= review_max_m or active_ratio <= review_max_z:
            new_dec = "review"
        else:
            new_dec = "dynamic"

        aid = r.get("asset_id")
        fn = r.get("filename")
        ckpt.update_decision(aid or fn, new_dec)
        restored_counts[new_dec] += 1

    print("🔄 Restored Checkpoint Classifications from Raw Motion Scores:")
    print(f"   • Static  : {restored_counts['static']}")
    print(f"   • Review  : {restored_counts['review']}")
    print(f"   • Dynamic : {restored_counts['dynamic']}")
    print()
    return sum(restored_counts.values())


def pull_albums_to_checkpoint(
    client: ImmichClient,
    ckpt: Checkpoint,
    dry_run: bool = False,
    sync_tags: bool = True,
):
    """Pulls classification changes from Immich albums back into local SQLite database."""
    print("📥 Pulling album state from Immich into local SQLite database...")
    albums = client.get_albums()
    albums_by_name = {a.get("albumName"): a for a in albums}

    static_name = os.environ.get("ALBUM_STATIC", "[Static] Videos")
    review_name = os.environ.get("ALBUM_REVIEW", "[Review] Videos")
    dynamic_name = os.environ.get("ALBUM_DYNAMIC", "[Dynamic] Videos")

    static_album = albums_by_name.get(static_name)
    review_album = albums_by_name.get(review_name)
    dynamic_album = albums_by_name.get(dynamic_name)

    if not static_album and not review_album and not dynamic_album:
        print(f"❌ Error: None of the classification albums ('{static_name}', '{review_name}', '{dynamic_name}') exist in Immich.")
        print("   Did you run `immich-static sync` to push classifications and create the albums first?")
        print("   Aborting pull to protect local database.")
        return

    static_ids = client.get_album_assets(static_album["id"]) if static_album else set()
    review_ids = client.get_album_assets(review_album["id"]) if review_album else set()
    dynamic_ids = client.get_album_assets(dynamic_album["id"]) if dynamic_album else set()

    total_album_assets = len(static_ids) + len(review_ids) + len(dynamic_ids)
    print(f"   • Album '{static_name}': {len(static_ids)} assets")
    print(f"   • Album '{review_name}': {len(review_ids)} assets")
    print(f"   • Album '{dynamic_name}': {len(dynamic_ids)} assets")
    print()

    if total_album_assets == 0:
        print("⚠️  Warning: The albums in Immich currently contain 0 assets.")
        print("   Did you populate the albums first using `immich-static sync`?")
        print("   Aborting pull to avoid accidentally clearing local classifications.")
        return

    all_rows = ckpt.all_rows()
    if not all_rows:
        print("⚠️  Checkpoint database is empty. No local records to update.")
        return

    changes: Dict[str, List[Tuple[str, str, str]]] = {
        "static_to_dynamic": [],
        "static_to_review": [],
        "review_to_static": [],
        "review_to_dynamic": [],
        "dynamic_to_static": [],
        "dynamic_to_review": [],
    }

    updated_count = 0

    for r in all_rows:
        aid = r.get("asset_id")
        old_decision = r.get("decision", "")
        if not aid:
            continue

        target_decision = None
        if aid in static_ids:
            target_decision = "static"
        elif aid in review_ids:
            target_decision = "review"
        elif aid in dynamic_ids:
            target_decision = "dynamic"
        elif old_decision == "static" and static_album and len(static_ids) > 0 and aid not in static_ids:
            target_decision = "dynamic"

        if target_decision and target_decision != old_decision:
            change_key = f"{old_decision}_to_{target_decision}"
            fname = r.get("original_file_name") or Path(r.get("filename", "")).name
            if change_key in changes:
                changes[change_key].append((aid, fname, target_decision))
            else:
                changes.setdefault(change_key, []).append((aid, fname, target_decision))

            if not dry_run:
                ckpt.update_decision(aid, target_decision)
            updated_count += 1

    print("🔄 Synchronization Summary:")
    has_any_change = False
    for change_key, items in changes.items():
        if items:
            has_any_change = True
            parts = change_key.split("_to_")
            from_dec, to_dec = parts[0], parts[1]
            print(f"   • Reclassified {len(items):>3} videos: [{from_dec.upper()}] ➔ [{to_dec.upper()}]")
            for aid, fname, _ in items[:5]:
                print(f"       - {fname} ({aid[:8]}...)")
            if len(items) > 5:
                print(f"       ... and {len(items) - 5} more")

    if not has_any_change:
        print("   ✅ Local checkpoint is already 100% in sync with Immich albums. No changes needed.")
    else:
        status_msg = "Would update" if dry_run else "Successfully updated"
        print(f"\n   🎉 {status_msg} {updated_count} record(s) in SQLite checkpoint.")

    # Re-sync tags if requested
    if sync_tags and updated_count > 0 and not dry_run:
        print("\n🏷  Updating Immich tags to match new album classifications...")
        existing_tags = client.get_tags()
        tags_by_name = {t.get("name"): t for t in existing_tags}
        tag_name_map = {
            "static": os.environ.get("TAG_STATIC", "video:static"),
            "review": os.environ.get("TAG_REVIEW", "video:review"),
            "dynamic": os.environ.get("TAG_DYNAMIC", "video:dynamic"),
        }
        for change_key, items in changes.items():
            if not items:
                continue
            parts = change_key.split("_to_")
            from_dec, to_dec = parts[0], parts[1]
            old_tag_name = tag_name_map.get(from_dec)
            new_tag_name = tag_name_map.get(to_dec)
            old_tag_obj = tags_by_name.get(old_tag_name)
            new_tag_obj = tags_by_name.get(new_tag_name)
            aids = [item[0] for item in items]
            if old_tag_obj:
                client.untag_assets(old_tag_obj["id"], aids)
            if not new_tag_obj and new_tag_name:
                new_tag_obj = client.create_tag(new_tag_name)
                tags_by_name[new_tag_name] = new_tag_obj
            if new_tag_obj:
                client.tag_assets(new_tag_obj["id"], aids)
        print("   ✅ Immich tags updated.")
    print()


def display_storage_stats(
    client: Optional[ImmichClient],
    ckpt: Optional[Checkpoint] = None,
):
    """Calculates and displays a storage breakdown and estimated savings across video albums."""
    print("📊 Calculating video storage and duration metrics...")

    stats: Dict[str, Dict[str, Any]] = {
        "static": {"count": 0, "size_bytes": 0, "duration_s": 0.0},
        "review": {"count": 0, "size_bytes": 0, "duration_s": 0.0},
        "dynamic": {"count": 0, "size_bytes": 0, "duration_s": 0.0},
        "other": {"count": 0, "size_bytes": 0, "duration_s": 0.0},
    }
    extracted_stats = {"count": 0, "size_bytes": 0}

    if client:
        albums = client.get_albums()
        album_map = {a.get("albumName"): a for a in albums if isinstance(a, dict)}

        static_album = album_map.get("[Static] Videos")
        review_album = album_map.get("[Review] Videos")
        dynamic_album = album_map.get("[Dynamic] Videos")
        extracted_album = album_map.get("[Extracted] Photos")

        static_ids = client.get_album_assets(static_album["id"]) if static_album else set()
        review_ids = client.get_album_assets(review_album["id"]) if review_album else set()
        dynamic_ids = client.get_album_assets(dynamic_album["id"]) if dynamic_album else set()
        extracted_ids = client.get_album_assets(extracted_album["id"]) if extracted_album else set()

        videos = client.get_all_video_assets()

        for v in videos:
            aid = v.get("id")
            exif = v.get("exifInfo") or {}
            size_bytes = int(exif.get("fileSizeInByte") or v.get("size") or 0)

            dur_s = 0.0
            raw_dur = v.get("duration")
            if isinstance(raw_dur, (int, float)):
                dur_s = float(raw_dur) / 1000.0 if raw_dur > 0 else 0.0
            elif isinstance(raw_dur, str) and ":" in raw_dur:
                try:
                    parts = [float(p) for p in raw_dur.split(":")]
                    dur_s = sum(p * (60 ** i) for i, p in enumerate(reversed(parts)))
                except Exception:
                    dur_s = 0.0
            elif raw_dur:
                try:
                    dur_s = float(raw_dur) / 1000.0 if float(raw_dur) > 500 else float(raw_dur)
                except Exception:
                    dur_s = 0.0

            if aid in static_ids:
                cat = "static"
            elif aid in review_ids:
                cat = "review"
            elif aid in dynamic_ids:
                cat = "dynamic"
            else:
                cat = "other"

            stats[cat]["count"] += 1
            stats[cat]["size_bytes"] += size_bytes
            stats[cat]["duration_s"] += dur_s

        if extracted_ids:
            extracted_stats["count"] = len(extracted_ids)
            extracted_stats["size_bytes"] = client.get_assets_total_size(extracted_ids)

    elif ckpt:
        all_rows = ckpt.all_rows()
        for r in all_rows:
            dec = r.get("decision", "other")
            if dec not in stats:
                dec = "other"

            size_bytes = 0
            fn = r.get("filename")
            if fn:
                p = Path(fn)
                if p.is_file():
                    try:
                        size_bytes = p.stat().st_size
                    except Exception:
                        pass

            dur_s = 0.0
            try:
                dur_s = float(r.get("duration_s") or 0.0)
            except (ValueError, TypeError):
                pass

            stats[dec]["count"] += 1
            stats[dec]["size_bytes"] += size_bytes
            stats[dec]["duration_s"] += dur_s

        ext_dir = Path(os.environ.get("EXTRACT_OUTPUT_DIR", "./extracted_frames")).resolve()
        if ext_dir.exists():
            ext_files = [p for p in ext_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
            extracted_stats["count"] = len(ext_files)
            extracted_stats["size_bytes"] = sum(p.stat().st_size for p in ext_files)

    total_count = sum(s["count"] for s in stats.values())
    total_size = sum(s["size_bytes"] for s in stats.values())
    total_dur = sum(s["duration_s"] for s in stats.values())

    static_size = stats["static"]["size_bytes"]
    static_count = stats["static"]["count"]
    actual_frame_size = extracted_stats["size_bytes"] if extracted_stats["count"] > 0 else (static_count * 350 * 1024)
    potential_savings_bytes = max(0, static_size - actual_frame_size)
    potential_savings_pct = (potential_savings_bytes / static_size * 100) if static_size > 0 else 0.0

    print("\n" + "═" * 72)
    print("               📦 IMMICH VIDEO STORAGE & ALBUM BREAKDOWN")
    print("═" * 72)
    print(f" {'Category / Album':<22} | {'Count':<8} | {'Total Duration':<16} | {'Storage Space':<15}")
    print("─" * 72)

    categories = [
        ("static", "📁 [Static] Videos"),
        ("review", "📁 [Review] Videos"),
        ("dynamic", "📁 [Dynamic] Videos"),
    ]
    if stats["other"]["count"] > 0:
        categories.append(("other", "❓ Other / Errors"))

    for key, label in categories:
        c = stats[key]["count"]
        d = format_duration(stats[key]["duration_s"])
        s = format_bytes(stats[key]["size_bytes"])
        print(f" {label:<22} | {c:<8} | {d:<16} | {s:<15}")

    print("─" * 72)
    print(f" {'TOTAL':<22} | {total_count:<8} | {format_duration(total_dur):<16} | {format_bytes(total_size):<15}")
    print("═" * 72)

    if extracted_stats["count"] > 0:
        print(f" 🖼️  [Extracted] Photos  | {extracted_stats['count']:<8} | {'—':<16} | {format_bytes(extracted_stats['size_bytes']):<15}")
        print("─" * 72)

    if static_count > 0 and static_size > 0:
        print("\n💡 Storage Optimization Potential:")
        print(f"   • Static videos storage       : {format_bytes(static_size)} ({static_count} videos)")
        if extracted_stats["count"] > 0:
            print(f"   • Extracted still frames size : {format_bytes(extracted_stats['size_bytes'])} ({extracted_stats['count']} photos)")
        else:
            print(f"   • Extracted still frames size : ~{format_bytes(actual_frame_size)}")
        print(f"   • Estimated space savings     : {format_bytes(potential_savings_bytes)} ({potential_savings_pct:.1f}% reduction)")
    print()


def upload_extracted_frames(client: ImmichClient, output_dir: Path, dry_run: bool = False):
    """Uploads local extracted frames in output_dir to Immich and assigns tag & album."""
    if not output_dir.exists():
        print(f"⚠️  Extracted frames directory not found: {output_dir}")
        return

    images = sorted([p for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not images:
        print(f"⚠️  No images found in {output_dir} to upload.")
        return

    tag_name = os.environ.get("TAG_EXTRACTED", "video:extracted")
    if dry_run:
        print(f"🔍 [DRY-RUN] Would upload {len(images)} frames from {output_dir} and tag as '{tag_name}'")
        return

    print(f"☁️  Uploading {len(images)} extracted frames to Immich...")
    tags = client.get_tags()
    extracted_tag = next((t for t in tags if t.get("name") == tag_name), None)
    if not extracted_tag:
        extracted_tag = client.create_tag(tag_name)
    tag_id = extracted_tag["id"]

    uploaded_ids = []
    for img_path in images:
        res = client.upload_asset(img_path, created_at=img_path.stat().st_mtime)
        if res and "id" in res:
            aid = res["id"]
            status_str = res.get("status", "created")
            uploaded_ids.append(aid)
            print(f"   Uploaded: {img_path.name} -> Immich ID: {aid} ({status_str})")
        else:
            print(f"   Uploaded: {img_path.name} -> (Response: {res})")

    if uploaded_ids:
        client.tag_assets(tag_id, uploaded_ids)
        print(f"✅ Uploaded and tagged with '{tag_name}' ({len(uploaded_ids)} items)")

        album_name = "[Extracted] Photos"
        albums = client.get_albums()
        target_album = next((a for a in albums if a.get("albumName") == album_name), None)
        if not target_album:
            target_album = client.create_album(album_name)
        album_id = target_album["id"]
        client.add_assets_to_album(album_id, uploaded_ids)
        print(f"📁 Added {len(uploaded_ids)} images to Album '{album_name}'")

        album_asset_ids = client.get_album_assets(album_id)
        all_to_tag = list(set(uploaded_ids) | album_asset_ids)
        client.tag_assets(tag_id, all_to_tag)
        print(f"🏷  Verified tag '{tag_name}' across all {len(all_to_tag)} assets in '{album_name}'")


def _get_client_and_ckpt(args: Any, require_ckpt: bool = True, require_client: bool = True) -> Tuple[Optional[ImmichClient], Optional[Checkpoint]]:
    api_url = getattr(args, "api_url", None) or os.environ.get("IMMICH_API_URL")
    api_key = getattr(args, "api_key", None) or os.environ.get("IMMICH_API_KEY")
    db_path_str = getattr(args, "db_path", None) or os.environ.get("DB_PATH", "./" + CHECKPOINT_FILENAME)
    db_path = Path(db_path_str).resolve()

    ckpt = None
    if db_path.exists():
        ckpt = Checkpoint(db_path)
    elif require_ckpt:
        print(f"❌ Error: Checkpoint database not found at {db_path}")
        print("   Run `immich-static detect` first to analyze your video library.")
        sys.exit(1)

    client = None
    if api_url and api_key:
        client = ImmichClient(api_url, api_key)
        try:
            ver = client.ping()
            print(f"🌐 Connected to Immich Server (Version: {ver.get('major', '')}.{ver.get('minor', '')}.{ver.get('patch', '')})")
        except Exception as e:
            if require_client:
                print(f"❌ Failed to connect to Immich server at {api_url}: {e}")
                sys.exit(1)
            else:
                print(f"⚠️  Immich server connection failed ({e}), proceeding with local data only.")
                client = None
    elif require_client:
        print("❌ Error: Missing Immich API configuration.")
        print("   Provide --api-url and --api-key or set IMMICH_API_URL and IMMICH_API_KEY in .env")
        sys.exit(1)

    return client, ckpt


def run_sync(args: Any):
    """Executes the `immich-static sync` command."""
    client, ckpt = _get_client_and_ckpt(args, require_ckpt=True, require_client=True)
    dry_run = getattr(args, "dry_run", False)
    sync_tags = not getattr(args, "no_tags", False)
    sync_albums = not getattr(args, "no_albums", False)
    include_dynamic = getattr(args, "include_dynamic", True)

    if dry_run:
        print("🔎 Running in DRY-RUN mode. No write changes will be made.\n")

    sync_immich(
        client=client,
        ckpt=ckpt,
        dry_run=dry_run,
        sync_tags=sync_tags,
        sync_albums=sync_albums,
        include_dynamic=include_dynamic,
    )
    display_storage_stats(client=client, ckpt=ckpt)

    if getattr(args, "upload_extracted", False) and client:
        ext_dir = Path(os.environ.get("EXTRACT_OUTPUT_DIR", "./extracted_frames")).resolve()
        upload_extracted_frames(client, ext_dir, dry_run=dry_run)

    print("✨ Complete!")


def run_pull(args: Any):
    """Executes the `immich-static pull` command."""
    client, ckpt = _get_client_and_ckpt(args, require_ckpt=True, require_client=True)
    dry_run = getattr(args, "dry_run", False)
    sync_tags = not getattr(args, "no_tag_sync_on_pull", False)

    if dry_run:
        print("🔎 Running in DRY-RUN mode. No write changes will be made.\n")

    pull_albums_to_checkpoint(
        client=client,
        ckpt=ckpt,
        dry_run=dry_run,
        sync_tags=sync_tags,
    )
    print("✨ Complete!")


def run_stats(args: Any):
    """Executes the `immich-static stats` command."""
    client, ckpt = _get_client_and_ckpt(args, require_ckpt=False, require_client=False)
    display_storage_stats(client=client, ckpt=ckpt)
    print("✨ Complete!")


def run_restore(args: Any):
    """Executes the `immich-static restore` command."""
    _, ckpt = _get_client_and_ckpt(args, require_ckpt=True, require_client=False)
    sensitivity = getattr(args, "sensitivity", "medium") or "medium"
    restore_checkpoint_decisions(ckpt, sensitivity=sensitivity)
    display_storage_stats(client=None, ckpt=ckpt)
    print("✨ Complete!")
