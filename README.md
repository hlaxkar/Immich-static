# 📸 Immich Static Video Toolkit (`immich-static`)

A high-performance toolkit designed to scan large Immich libraries (60,000+ assets) via **direct filesystem mount**, detect **static / low-motion videos** (screen recordings, static slides, accidental burst videos, live photos), classify them into Immich albums, and extract crisp, sharpness-optimized still frames.

---

## ⚡ Key Features

1. **Direct Mount Speed (0 Network Overhead for Video Decoding)**:
   - Reads directly from your local/mounted Immich storage volume (`/path/to/immich/library`).
   - Uses sparse keyframe sampling and downscaled motion grids for high processing throughput across multi-core CPUs.
2. **Multi-Frame Motion & Zone Analysis**:
   - Analyzes global motion and $4\times4$ localized grid zones to classify videos into `static`, `review`, and `dynamic`.
3. **Resumable SQLite Checkpoint**:
   - Maintains a concurrent WAL SQLite database (`detection_checkpoint.sqlite`) so you can pause, interrupt (Ctrl+C), or resume at any time without re-analyzing processed assets.
4. **Immich Album Sync Bridge**:
   - Resolves local file paths to Immich Asset IDs via lightweight metadata API calls.
   - Automatically populates `[Static] Videos`, `[Review] Videos`, and `[Dynamic] Videos` albums in your Immich instance.
5. **Sharpness-Aware Frame Extractor**:
   - Uses Laplacian variance ($Var(\nabla^2 f)$) across sampled frames to select and extract the single sharpest, least-blurry photo.
6. **Continuous Incremental Watcher**:
   - Runs in the background to automatically detect and classify future video uploads.

---

## 🛠 Prerequisites

- Python 3.9+
- `ffmpeg` & `ffprobe` installed on system
- Python dependencies:
  ```bash
  pip install opencv-python-headless numpy tqdm
  ```

---

## 🚀 Step-by-Step Usage Workflow

### 1. Step 1: Scan & Classify Mounted Immich Library (Safe / Local Only)
Run detection against your mounted library folder:

```bash
# Scan and classify (saves to SQLite locally, makes NO changes to Immich)
python detect.py

# Useful flags:
#   --sync               (Optional: automatically sync tags & albums immediately after detection)
#   --workers 8          (Set custom worker count)
#   --sensitivity high   (low | medium | high)
#   --report             (Print confidence & motion score breakdown)
#   --export-csv [PATH]  (Optional: export results to CSV spreadsheet)
```

---

### 2. Step 2: Sync Results to Immich Tags & Albums
Tag every video in Immich with its classification (**`video:static`**, **`video:review`**, **`video:dynamic`**) and organize them into albums:

```bash
# 1. Preview changes safely without modifying Immich:
python immich_sync.py --dry-run

# 2. Live sync (applies tags and populates albums in Immich):
python immich_sync.py

# 3. Optional: Sync tags/albums AND upload local extracted frames in one step:
python immich_sync.py --upload-extracted
```

---

### 3. Step 3: Extract Best Frames & Upload with "video:extracted" Tag
Extract the sharpest frame from all videos classified as `static` directly from the checkpoint database:

```bash
# 1. Extract frames locally (preserves original video creation timestamp and EXIF):
python extract.py

# 2. Extract AND upload directly to Immich with 'video:extracted' tag:
python extract.py --upload

# 3. If you extracted locally earlier, upload them now:
python extract.py --upload
# or:
python immich_sync.py --upload-extracted
```

---

### 4. Step 4: Continuous Monitoring for Future Uploads
To automatically classify and tag new videos uploaded to your Immich instance in the background:

```bash
python watch.py
```

---

## 📁 File Structure

- `core.py`: Shared core engine (FFmpeg sampling, motion scoring, Laplacian sharpness evaluation, SQLite checkpointing).
- `detect.py`: Multi-threaded library scanner and motion classifier.
- `extract.py`: Sharpness-optimized frame extractor (DB-driven or directory-driven).
- `immich_sync.py`: Immich REST API bridge for asset ID resolution and album management.
- `watch.py`: Background daemon for incremental scanning.
- `test_suite.py`: Synthetic test video generator and validation suite.
