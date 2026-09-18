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

    # 1. Generate Static Video (a solid color slide with text for 3 seconds)
    print("🎬 Generating synthetic static video...")
    cmd_static = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "color=c=navy:s=640x360:d=3:r=25",
        "-vf", "drawtext=text='STATIC SLIDE':fontcolor=white:fontsize=32:x=(w-text_w)/2:y=(h-text_h)/2",
        "-pix_fmt", "yuv420p",
        str(static_mp4)
    ]
    subprocess.run(cmd_static, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2. Generate High-Motion Dynamic Video (rapid scrolling full-frame pattern for 3 seconds)
    print("🎬 Generating synthetic dynamic video...")
    cmd_dynamic = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "smptebars=size=640x360:rate=25:duration=3",
        "-vf", "scroll=horizontal=0.2",
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
        assert meta_static["width"] == 640
        assert meta_static["height"] == 360
        assert abs(meta_static["duration"] - 3.0) < 0.5
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

        # Verify asset was classified as static and tagged/album added
        assert ckpt.is_done(test_asset_id) is True
        row = ckpt.get(test_asset_id)
        assert row["decision"] == "static"
        assert test_asset_id in mock_client.tagged
        assert test_asset_id in mock_client.album_added
        print(f"   • Asset processed: {test_asset_id} -> STATIC, tagged and added to album!")

        # Shutdown test server
        server.shutdown()
        server.server_close()
        print("✅ Webhook Server verified end-to-end!")

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
