"""
immich_static.test_suite — Automated synthetic test suite for verifying immich-static components.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any

from immich_static.core import (
    SENSITIVITY_PRESETS,
    Checkpoint,
    detect_video,
    extract_one_frame,
    get_video_metadata,
)

TEST_DIR = Path("./tmp_test_env").resolve()


def generate_synthetic_videos():
    """Generates synthetic test videos using ffmpeg."""
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    static_mp4 = TEST_DIR / "test_static.mp4"
    dynamic_mp4 = TEST_DIR / "test_dynamic.mp4"

    # 1. Generate Static Video (representative 9:16 vertical slide with audio track for 10 seconds)
    print("🎬 Generating synthetic static video...")
    cmd_static = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "color=c=navy:s=360x640:d=10:r=25",
        "-f", "lavfi",
        "-i", "anullsrc=r=44100:cl=stereo",
        "-vf", "drawtext=text='STATIC SLIDE':fontcolor=white:fontsize=32:x=(w-text_w)/2:y=(h-text_h)/2",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        "-pix_fmt", "yuv420p",
        str(static_mp4)
    ]
    subprocess.run(cmd_static, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2. Generate High-Motion Dynamic Video (active motion with moving elements and changing patterns)
    print("🎬 Generating synthetic dynamic video...")
    cmd_dynamic = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "testsrc2=size=640x360:rate=25:duration=5",
        "-pix_fmt", "yuv420p",
        str(dynamic_mp4)
    ]
    subprocess.run(cmd_dynamic, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return static_mp4, dynamic_mp4


def run_tests():
    """Runs all end-to-end verification tests."""
    print("========================================")
    print("🧪 Running Immich-Static Test Suite")
    print("========================================")

    static_vid, dynamic_vid = generate_synthetic_videos()
    thresholds = SENSITIVITY_PRESETS["medium"]

    try:
        # Test 1: Metadata Extraction
        print("\n--- Test 1: Video Metadata Probe ---")
        meta_static = get_video_metadata(static_vid)
        print(f"Static video: {meta_static['width']}x{meta_static['height']}, {meta_static['duration']}s, {meta_static['fps']}fps")
        assert meta_static["width"] == 360
        assert meta_static["height"] == 640
        assert abs(meta_static["duration"] - 10.0) < 0.5
        print("✅ Metadata probe passed!")

        # Test 2: Detection on Static Video
        print("\n--- Test 2: Static Video Detection ---")
        res_static = detect_video(static_vid, thresholds)
        print(f"Static video result: decision={res_static['decision']}, motion={res_static['global_motion_score']}, zones={res_static['active_zone_ratio']}")
        assert res_static["decision"] == "static", f"Expected 'static', got {res_static['decision']}"
        print("✅ Static video correctly identified!")

        # Test 3: Detection on Dynamic Video
        print("\n--- Test 3: Dynamic Video Detection ---")
        res_dynamic = detect_video(dynamic_vid, thresholds)
        print(f"Dynamic video result: decision={res_dynamic['decision']}, motion={res_dynamic['global_motion_score']}, zones={res_dynamic['active_zone_ratio']}")
        assert res_dynamic["decision"] == "dynamic", f"Expected 'dynamic', got {res_dynamic['decision']}"
        print("✅ Dynamic video correctly identified!")

        # Test 4: SQLite Checkpointing
        print("\n--- Test 4: SQLite Resumable Checkpoint ---")
        db_file = TEST_DIR / "test_ckpt.sqlite"
        ckpt = Checkpoint(db_file)
        ckpt.clear()
        ckpt.record(str(static_vid), res_static)
        ckpt.record(str(dynamic_vid), res_dynamic)

        assert ckpt.is_done(str(static_vid)) is True
        assert ckpt.is_done(str(dynamic_vid)) is True
        assert len(ckpt.all_rows()) == 2
        print("✅ Checkpoint persistence verified!")

        # Test 5: Laplacian Frame Extraction
        print("\n--- Test 5: Laplacian Sharpness Frame Extraction ---")
        out_frame = TEST_DIR / "extracted_static.jpg"
        extract_res = extract_one_frame(static_vid, out_frame, fmt="jpg", quality=95)
        print(f"Extraction result: status={extract_res['status']}, sharpness={extract_res.get('sharpness', '?')}, timestamp={extract_res.get('timestamp_s', '?')}s")
        assert extract_res["status"] == "ok"
        assert out_frame.exists() and out_frame.stat().st_size > 0
        print("✅ Best frame extracted successfully!")

        # Test 6: Real-time Webhook Server & Async Dispatch
        print("\n--- Test 6: Real-time Webhook Server & Async Dispatch ---")
        import json
        import threading
        import time
        import urllib.request
        from immich_static.server import WebhookRequestHandler, WebhookServer, _worker_loop

        class MockImmichClient:
            def __init__(self):
                self.tagged = []
                self.album_added = []
            def ping(self):
                return {"major": 3, "minor": 2, "patch": 0}
            def get_asset_info(self, aid):
                return {
                    "id": aid,
                    "type": "VIDEO",
                    "originalPath": str(static_vid),
                    "originalFileName": static_vid.name,
                }
            def get_tags(self):
                return [{"id": "tag-static-id", "name": "video:static"}]
            def create_tag(self, name):
                return {"id": f"tag-{name}", "name": name}
            def tag_assets(self, tid, aids):
                self.tagged.extend(aids)
            def get_albums(self):
                return [{"id": "album-static-id", "albumName": "[Static] Videos"}]
            def create_album(self, name):
                return {"id": f"album-{name}", "albumName": name}
            def add_assets_to_album(self, aid, aids):
                self.album_added.extend(aids)
                return len(aids), aids
            def find_parent_motion_photo(self, vid, orig_name=None):
                return None

        mock_client = MockImmichClient()
        server = WebhookServer(
            ("127.0.0.1", 0),
            WebhookRequestHandler,
            client=mock_client,
            ckpt=ckpt,
            thresholds=thresholds,
            folder=TEST_DIR,
            remote_prefix=None,
            local_prefix=None,
            secret="test_secret_123",
            include_dynamic=True,
            extract_frame=False,
            dry_run=False,
        )
        port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        worker_thread = threading.Thread(target=_worker_loop, args=(server, 1), daemon=True)
        worker_thread.start()

        # 6a: Health check
        req_health = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        with urllib.request.urlopen(req_health, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            assert data["status"] == "ok"
            print("   • Health check endpoint: OK (200)")

        # 6b: Auth rejection without secret
        req_no_auth = urllib.request.Request(
            f"http://127.0.0.1:{port}/webhook",
            data=json.dumps({"id": "test-1"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            urllib.request.urlopen(req_no_auth, timeout=5)
            assert False, "Expected HTTP 401 Unauthorized"
        except urllib.error.HTTPError as e:
            assert e.code == 401
            print("   • Auth rejection without secret: OK (401)")

        # 6c: Ignored non-video payload
        req_image = urllib.request.Request(
            f"http://127.0.0.1:{port}/webhook",
            data=json.dumps({"id": "img-1", "type": "IMAGE"}).encode(),
            headers={"Content-Type": "application/json", "X-Webhook-Secret": "test_secret_123"},
            method="POST"
        )
        with urllib.request.urlopen(req_image, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            assert data["status"] == "ignored"
            print("   • Non-video filter: OK (ignored)")

        # 6d: Enqueue and process video asset
        test_asset_id = "test-video-asset-uuid-123"
        req_video = urllib.request.Request(
            f"http://127.0.0.1:{port}/webhook",
            data=json.dumps({
                "id": test_asset_id,
                "type": "VIDEO",
                "originalPath": str(static_vid),
                "originalFileName": static_vid.name,
            }).encode(),
            headers={"Content-Type": "application/json", "X-Webhook-Secret": "test_secret_123"},
            method="POST"
        )
        with urllib.request.urlopen(req_video, timeout=5) as resp:
            assert resp.status == 202
            data = json.loads(resp.read().decode())
            assert data["status"] == "queued"
            print("   • Webhook accepted: OK (202 Accepted)")

        # Wait for queue to drain
        server.worker_queue.join()

        # Verify asset was classified as static and added to album
        assert ckpt.is_done(test_asset_id) is True
        row = ckpt.get(test_asset_id)
        assert row["decision"] == "static"
        assert test_asset_id in mock_client.album_added
        print(f"   • Asset processed: {test_asset_id} -> STATIC, added to album!")

        # Shutdown test server
        server.shutdown()
        server.server_close()
        print("✅ Webhook Server verified end-to-end!")

        # Test 7: Short Clip Handling (< 1.5s)
        print("\n--- Test 7: Short Clip Handling (< 1.5s) ---")
        short_vid = TEST_DIR / "test_short.mp4"
        cmd_short = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "color=c=navy:s=360x640:d=0.8:r=25",
            "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=stereo",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-shortest",
            "-pix_fmt", "yuv420p",
            str(short_vid),
        ]
        subprocess.run(cmd_short, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        res_short = detect_video(short_vid, thresholds)
        print(f"Short video (0.8s) detection: decision={res_short['decision']}, sampled={res_short['frames_sampled']}")
        assert res_short["frames_sampled"] >= 2, "Expected at least 2 sampled frames"
        assert res_short["decision"] == "static", f"Expected 'static', got {res_short['decision']}"

        out_short_frame = TEST_DIR / "extracted_short.jpg"
        ext_short_res = extract_one_frame(short_vid, out_short_frame)
        assert ext_short_res["status"] == "ok", f"Expected ok extraction, got {ext_short_res}"
        assert out_short_frame.exists() and out_short_frame.stat().st_size > 0
        print("✅ Short clip detection and frame extraction passed!")

        # Test 8: EXIF GPS & UTC Datetime Parsing
        print("\n--- Test 8: EXIF GPS & UTC Datetime Parsing ---")
        from immich_static.exif import deg_to_dms, parse_iso_datetime
        import datetime

        # Test DMS rollover
        d, m, s = deg_to_dms(12.99999999)
        assert s < 60.0, f"Seconds overflowed: {s}"
        assert d == 13.0 and m == 0.0 and s == 0.0, f"Unexpected rollover: {d}, {m}, {s}"

        # Test UTC Z parsing
        _, offset, dt_obj = parse_iso_datetime("2024-09-26T14:30:00.000Z")
        assert dt_obj is not None
        assert dt_obj.tzinfo == datetime.timezone.utc, "Expected UTC timezone-aware datetime"
        assert offset == "+00:00", f"Expected +00:00 offset, got {offset}"
        print("✅ EXIF GPS & UTC Datetime precision passed!")

        # Test 9: Pure Visual Motion Verification
        print("\n--- Test 9: Pure Visual Motion Verification ---")
        res_pure = detect_video(static_vid, thresholds)
        assert res_pure["decision"] == "static"
        print("✅ Visual motion classification passed!")

        # Test 10: Tag-Authoritative Pull & Conflict Resolution
        # Test 10: Album-Authoritative Pull & Conflict Resolution (User Albums Safe)
        print("\n--- Test 10: Album-Authoritative Pull & Conflict Resolution (User Albums Safe) ---")
        from immich_static.sync import pull_immich_to_checkpoint

        class MockAlbumImmichClient:
            def __init__(self):
                self.albums = [
                    {"id": "a-static", "albumName": "[Static] Videos"},
                    {"id": "a-review", "albumName": "[Review] Videos"},
                    {"id": "a-dynamic", "albumName": "[Dynamic] Videos"},
                    {"id": "a-pets", "albumName": "Pets"},
                ]
                self.album_to_assets = {
                    "a-static": {"aid-static", "aid-conflict"},
                    "a-review": set(),
                    "a-dynamic": {"aid-dynamic", "aid-conflict"},
                    "a-pets": {"aid-static", "aid-conflict", "aid-dynamic"},
                }
                self.removed_album = []
                self.added_album = []

            def get_albums(self): return self.albums
            def get_album_assets(self, aid): return set(self.album_to_assets.get(aid, set()))
            def remove_assets_from_album(self, aid, aids):
                self.removed_album.append((aid, aids))
                for a in aids:
                    self.album_to_assets.get(aid, set()).discard(a)
                return len(aids)
            def add_assets_to_album(self, aid, aids):
                self.added_album.append((aid, aids))
                for a in aids:
                    self.album_to_assets.setdefault(aid, set()).add(a)
                return len(aids), aids

        test_ckpt_file = TEST_DIR / "test_album_pull.sqlite"
        t_ckpt = Checkpoint(test_ckpt_file)
        t_ckpt.clear()

        # Existing synced static video
        t_ckpt.record("aid-static", {"asset_id": "aid-static", "decision": "static", "synced_album": "[Static] Videos", "synced_at": "2026-10-01"})
        # Conflicted video (present in both Static & Dynamic, PLUS user album 'Pets')
        t_ckpt.record("aid-conflict", {"asset_id": "aid-conflict", "decision": "static", "synced_album": "[Static] Videos", "synced_at": "2026-10-01"})
        # Newly detected video waiting for initial sync
        t_ckpt.record("aid-newly-detected", {"asset_id": "aid-newly-detected", "decision": "static", "synced_album": None, "synced_at": None})

        mock_album_client = MockAlbumImmichClient()
        pull_immich_to_checkpoint(mock_album_client, t_ckpt, dry_run=False)

        assert t_ckpt.get("aid-static")["decision"] == "static"
        assert t_ckpt.get("aid-conflict")["decision"] == "review", "Conflicted asset should resolve to review"
        # Removed from static and dynamic classification albums
        assert ("a-static", ["aid-conflict"]) in mock_album_client.removed_album
        assert ("a-dynamic", ["aid-conflict"]) in mock_album_client.removed_album
        # Added to review album
        assert ("a-review", ["aid-conflict"]) in mock_album_client.added_album
        # User album 'Pets' MUST NOT BE TOUCHED
        for aid_rem, _ in mock_album_client.removed_album:
            assert aid_rem != "a-pets", "User album 'Pets' must never be modified by conflict resolution!"
        assert "aid-conflict" in mock_album_client.album_to_assets["a-pets"], "User album 'Pets' must still contain the video!"
        assert t_ckpt.get("aid-newly-detected")["decision"] == "static", "Newly detected unsynced video must remain untouched"
        print("✅ Album-authoritative pull, conflict resolution, and user album safety verified!")

        # Test 11: On-Demand Webhook Pull Endpoint (/webhook/pull)
        print("\n--- Test 11: On-Demand Webhook Pull Endpoint (/webhook/pull) ---")
        pull_server = WebhookServer(
            ("127.0.0.1", 0),
            WebhookRequestHandler,
            client=mock_album_client,
            ckpt=t_ckpt,
            thresholds=thresholds,
            folder=TEST_DIR,
            remote_prefix=None,
            local_prefix=None,
            secret="test_secret_123",
            dry_run=False,
        )
        pull_port = pull_server.server_address[1]
        pull_server_thread = threading.Thread(target=pull_server.serve_forever, daemon=True)
        pull_server_thread.start()
        pull_worker_thread = threading.Thread(target=_worker_loop, args=(pull_server, 1), daemon=True)
        pull_worker_thread.start()

        # Simulate user moving aid-static from [Static] Videos to [Dynamic] Videos in Immich UI
        mock_album_client.album_to_assets["a-static"].discard("aid-static")
        mock_album_client.album_to_assets["a-dynamic"].add("aid-static")

        req_pull = urllib.request.Request(
            f"http://127.0.0.1:{pull_port}/webhook/pull",
            headers={"Content-Type": "application/json", "X-Webhook-Secret": "test_secret_123"},
            data=b"{}",
            method="POST"
        )
        with urllib.request.urlopen(req_pull, timeout=5) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data["status"] == "ok"
            print("   • On-demand /webhook/pull accepted: OK (200 OK)")

        # Verify aid-static was updated to dynamic in SQLite
        assert t_ckpt.get("aid-static")["decision"] == "dynamic"
        assert t_ckpt.get("aid-static")["user_curated"] == 1
        print("   • On-demand album pull synced correctly: [STATIC] ➔ [DYNAMIC] in SQLite!")

        pull_server.shutdown()
        pull_server.server_close()
        print("✅ On-demand /webhook/pull verified end-to-end!")

        print("\n========================================")
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("========================================")
    finally:
        cleanup()


def cleanup():
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)


def run_test_cmd(args: Any = None):
    run_tests()


if __name__ == "__main__":
    run_tests()
