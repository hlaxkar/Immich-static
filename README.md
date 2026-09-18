# 📸 Immich Static Video Toolkit (`immich-static`)

An API-first, zero-download toolkit and CLI for Immich. It scans video libraries via **direct filesystem mount**, detects **static / low-motion videos** (slides, screen recordings, accidental burst videos, live photos), organizes them into Immich tags and albums via REST API, and extracts sharpness-optimized still frames.

---

## ⚡ Key Features

1. **Zero-Download Speed (Local Mount Data Plane)**:
   - Analyzes video frames directly on your mounted Immich storage volume (`/path/to/immich/library`) using multi-core OpenCV / FFmpeg.
2. **Multi-Frame Motion & Zone Analysis**:
   - Analyzes global motion and $4\times4$ localized grid zones to classify videos into `static`, `review`, and `dynamic`.
3. **Unified CLI Architecture**:
   - Single command `immich-static` with subcommands (`detect`, `sync`, `stats`, `pull`, `extract`, `watch`, `test`).
4. **Resumable SQLite Checkpoint**:
   - Thread-safe WAL SQLite database (`detection_checkpoint.sqlite`) allowing instant pause, interrupt (Ctrl+C), and resume.
5. **Bidirectional Immich Sync**:
   - Automatically populates `video:static`, `video:review`, and `video:dynamic` tags and albums in Immich.
   - Allows manual UI review in Immich and pulls changes back with `immich-static pull`.
6. **Sharpness-Aware Still Extractor**:
   - Uses Laplacian variance ($Var(\nabla^2 f)$) across sampled frames to select and extract the single sharpest photo with EXIF metadata preserved.
7. **Continuous Background Watcher**:
   - Background daemon (`immich-static watch`) for incremental scanning of new video uploads.

---

## 🛠 Installation & Prerequisites

- Python 3.9+
- `ffmpeg` & `ffprobe` installed on system (`sudo apt install ffmpeg`)

### Install via pip:
```bash
# Clone the repository
git clone https://github.com/yourusername/immich-static.git
cd immich-static

# Install locally in editable/development mode:
pip install -e .

# Or standard install:
pip install .
```

---

## ⚙️ Configuration (`.env`)

Create a `.env` file in your workspace or set system environment variables:

```bash
IMMICH_API_URL=https://immich.yourdomain.com/api
IMMICH_API_KEY=your_immich_api_key_here
IMMICH_LIBRARY_PATH=/mnt/storage/immich/library

# Optional defaults:
SENSITIVITY=medium          # low | medium | high
WORKERS=16                  # Parallel detection workers
EXTRACT_OUTPUT_DIR=./extracted_frames
EXTRACT_FORMAT=jpg          # jpg | png
EXTRACT_QUALITY=95          # 1-100
```

---

## 🚀 CLI Commands & Workflow

### 1. Scan & Classify Mounted Immich Library (Local Only / Safe)
```bash
# Scan and save classifications into SQLite checkpoint (no changes to Immich):
immich-static detect

# Useful flags:
immich-static detect --sensitivity high
immich-static detect --workers 8
immich-static detect --report                 # Print motion score details
immich-static detect --export-csv results.csv # Export detection table to CSV
immich-static detect --sync                   # Auto-sync to Immich after scan
immich-static detect --dry-run                # Preview tag/album changes
```

---

### 2. View Storage Metrics & Savings Potential
```bash
immich-static stats
```

---

### 3. Sync Results to Immich Tags & Albums
```bash
# Preview changes safely without modifying Immich:
immich-static sync --dry-run

# Live sync (creates tags and populates [Static], [Review], [Dynamic] albums):
immich-static sync

# Sync and upload local extracted frames:
immich-static sync --upload-extracted
```

---

### 4. Review in Immich & Pull Corrections Back to Database
1. Open **`[Static] Videos`** album in your Immich web / mobile app.
2. Select any false positives and remove them or move them to **`[Dynamic] Videos`**.
3. Pull your changes back into the local SQLite database:

```bash
# Preview changes:
immich-static pull --dry-run

# Apply Immich album modifications to local checkpoint:
immich-static pull
```

---

### 5. Extract Sharpest Frames & Upload
```bash
# 1. Extract frames locally (preserves original video timestamps):
immich-static extract

# 2. Extract and upload directly to Immich with 'video:extracted' tag and '[Extracted] Photos' album:
immich-static extract --upload

# 3. Custom options:
immich-static extract --target-decision review --format png
```

---

### 6. Real-Time Webhook Daemon (Recommended for Immich v3+)
Instead of periodic disk polling, run `immich-static` as an event-driven webhook listener triggered instantly when videos are uploaded:

```bash
# Start the webhook server daemon:
immich-static serve --port 8080 --secret my_secure_token

# Or with custom options:
immich-static serve \
    --host 0.0.0.0 \
    --port 8080 \
    --secret my_secure_token \
    --sensitivity medium \
    --workers 4 \
    --extract # Auto-extract sharpest still frame for static videos
```

#### Configuring Immich Workflows (Immich v3.0.0+):
1. In the Immich web interface, navigate to **Administration** > **Workflows**.
2. Click **Create Workflow** and set Trigger to **Asset Created** / **Asset Upload**.
3. (Optional) Add a condition: `asset.type == 'VIDEO'`.
4. Add action: **Webhook**:
   - **Method**: `POST`
   - **URL**: `http://<YOUR_IMMICH_STATIC_HOST>:8080/webhook`
   - **Headers**:
     - `Content-Type`: `application/json`
     - `X-Webhook-Secret`: `my_secure_token`
5. Save and activate the workflow. New video uploads will be classified and tagged instantly!

#### Running in Background via systemd (Auto-Start on Boot)
You can install and run the webhook server as a background service with a single command without writing configuration files manually:

```bash
# Auto-detects Python environment, .env file, and installs as auto-starting systemd service:
immich-static service install --port 8080 --secret my_secure_token

# Preview what it will configure without installing:
immich-static service install --dry-run

# Manage the service:
immich-static service status    # Check daemon health & status
immich-static service logs      # Stream live journal logs
immich-static service restart   # Restart daemon
immich-static service stop      # Stop daemon
immich-static service uninstall # Remove service unit
```
*Note: By default, it installs as a **user-level service** (`systemctl --user`) and enables lingering (`loginctl enable-linger`), meaning it starts automatically on system boot **without requiring sudo/root permissions**! (Pass `--system` if you prefer system-wide `/etc/systemd/system`).*

---

### 7. Continuous Polling Watcher (Legacy Fallback)
For offline environments or older Immich versions (< v3.0.0) without Workflows:
```bash
immich-static watch --interval 300
```

---

### 8. Run Verification Test Suite
```bash
immich-static test
```

---

## 📦 Package Structure

```
immich-static/
├── pyproject.toml              # Modern build configuration (PEP 517/621)
├── setup.py                    # Backward compatibility for pip install -e .
├── requirements.txt            # Python dependencies
├── immich_static/              # Main Python package
│   ├── __init__.py             # Package version & top-level exports
│   ├── __main__.py             # Entrypoint for `python -m immich_static`
│   ├── cli.py                  # Main CLI argument parser and subcommand router
│   ├── core.py                 # Motion analysis, Laplacian sharpness, Checkpoint DB
│   ├── client.py               # Robust Immich REST API client
│   ├── detect.py               # Multithreaded library scanner & classifier
│   ├── sync.py                 # Tags/albums sync, pull, stats, and upload logic
│   ├── server.py               # Real-time webhook server & async worker queue
│   ├── extract.py              # Sharpness-aware still frame extractor
│   ├── watch.py                # Periodic polling background watcher
│   └── test_suite.py           # Synthetic verification tests
├── detect.py                   # Backward-compatibility root shim
├── immich_sync.py              # Backward-compatibility root shim
├── extract.py                  # Backward-compatibility root shim
├── watch.py                    # Backward-compatibility root shim
├── serve.py                    # Backward-compatibility root shim
└── test_suite.py               # Backward-compatibility root shim
```

---

## 📄 License
MIT License.
