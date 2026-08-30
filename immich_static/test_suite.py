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
        print(f"Static video meta: {meta_static}")
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
        print(f"Extraction result: {extract_res}")
        assert extract_res["status"] == "ok"
        assert out_frame.exists() and out_frame.stat().st_size > 0
        print("✅ Best frame extracted successfully!")

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
