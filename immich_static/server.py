"""
immich_static.server — Event-driven webhook server for real-time Immich video processing.
"""

import json
import os
import queue
import signal
import sys
import tempfile
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from immich_static.client import ImmichClient
from immich_static.core import (
    CHECKPOINT_FILENAME,
    ENV,
    LOG_FIELDS,
    SENSITIVITY_PRESETS,
    VIDEO_EXTENSIONS,
    Checkpoint,
    detect_video,
    extract_one_frame,
    resolve_local_video_path,
)
from immich_static.sync import sync_single_asset

_server_stop_event = threading.Event()
_start_time = time.time()


def resolve_asset_file_path(
    asset_info: Dict[str, Any],
    folder: Optional[Path] = None,
    remote_prefix: Optional[str] = None,
    local_prefix: Optional[str] = None,
) -> Optional[Path]:
    """
    Resolves the physical filesystem path for an Immich asset on the local machine.
    Handles Docker container path mapping and direct filesystem matching.
    """
    orig_path_str = asset_info.get("originalPath") or ""
    orig_name = asset_info.get("originalFileName") or (Path(orig_path_str).name if orig_path_str else "")

    # 1. Check mapped prefix (Docker host <-> container path translation)
    if orig_path_str and remote_prefix and local_prefix:
        norm_orig = os.path.normpath(orig_path_str)
        norm_remote = os.path.normpath(remote_prefix)
        if norm_orig.startswith(norm_remote):
            rel = os.path.relpath(norm_orig, norm_remote)
            candidate = Path(local_prefix) / rel
            if candidate.is_file():
                return candidate.resolve()

    # 2. Check direct path
    if orig_path_str:
        candidate = Path(orig_path_str)
        if candidate.is_file():
            return candidate.resolve()

    # 3. Check via resolve_local_video_path
    if folder and folder.is_dir() and orig_path_str:
        resolved = resolve_local_video_path(orig_path_str, folder)
        if resolved and resolved.is_file():
            return resolved.resolve()

    # 4. Check relative to folder (IMMICH_LIBRARY_PATH)
    if folder and folder.is_dir():
        if orig_path_str:
            candidate = folder / orig_path_str.lstrip("/")
            if candidate.is_file():
                return candidate.resolve()

            parts = Path(orig_path_str).parts
            for i in range(len(parts)):
                sub_candidate = folder / Path(*parts[i:])
                if sub_candidate.is_file():
                    return sub_candidate.resolve()

        # 5. Search under folder by filename if unique or recent
        if orig_name:
            direct_name_match = folder / orig_name
            if direct_name_match.is_file():
                return direct_name_match.resolve()

    return None


def extract_asset_info_from_payload(payload: Any) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Extracts assetId and metadata dictionary from various Immich workflow webhook payload structures.
    Supports nested workflow contexts (e.g. data.asset), direct AssetDTO, and wrapped actions.
    """
    if not isinstance(payload, dict):
        return None, {}

    # 1. Immich Workflows standard structure: data -> asset -> id
    data = payload.get("data")
    if isinstance(data, dict):
        asset = data.get("asset")
        if isinstance(asset, dict) and "id" in asset:
            return str(asset["id"]), asset
        if "id" in data and any(k in data for k in ("originalPath", "type", "originalFileName", "ownerId")):
            return str(data["id"]), data

    # 2. Wrapped directly in asset container: asset -> id
    asset = payload.get("asset")
    if isinstance(asset, dict) and "id" in asset:
        return str(asset["id"]), asset

    # 3. Direct asset DTO
    if "id" in payload and any(k in payload for k in ("originalPath", "type", "originalFileName", "ownerId")):
        return str(payload["id"]), payload

    # 4. Recursive search for candidate asset dictionary
    candidates = []
    def _search(d: Any, depth: int = 0):
        if depth > 4 or not isinstance(d, dict):
            return
        if "id" in d and isinstance(d["id"], (str, int)):
            score = 0
            if "type" in d: score += 3
            if "originalPath" in d: score += 3
            if "originalFileName" in d: score += 2
            if "ownerId" in d: score += 1
            candidates.append((score, str(d["id"]), d))
        for v in d.values():
            if isinstance(v, dict):
                _search(v, depth + 1)

    _search(payload)
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1], candidates[0][2]

    # 5. Fallback for raw ID strings
    if "assetId" in payload:
        return str(payload["assetId"]), payload
    if "id" in payload:
        return str(payload["id"]), payload

    return None, payload


class WebhookServer(ThreadingHTTPServer):
    """Threading HTTP server with worker queue and configuration context."""

    def __init__(
        self,
        server_address,
        handler_class,
        client: ImmichClient,
        ckpt: Checkpoint,
        thresholds: dict,
        folder: Optional[Path],
        remote_prefix: Optional[str],
        local_prefix: Optional[str],
        secret: Optional[str],
        include_dynamic: bool = True,
        extract_frame: bool = False,
        extract_dir: Optional[Path] = None,
        dry_run: bool = False,
    ):
        super().__init__(server_address, handler_class)
        self.client = client
        self.ckpt = ckpt
        self.thresholds = thresholds
        self.folder = folder
        self.remote_prefix = remote_prefix
        self.local_prefix = local_prefix
        self.secret = secret
        self.include_dynamic = include_dynamic
        self.extract_frame = extract_frame
        self.extract_dir = extract_dir
        self.dry_run = dry_run
        self.worker_queue: queue.Queue = queue.Queue()


def _log(msg: str):
    """Outputs a timestamped log line with immediate buffer flushing."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class WebhookRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Immich webhook callbacks and health status."""

    server: WebhookServer

    def log_message(self, format: str, *args: Any):
        # Override standard HTTP access log to use our custom formatter
        pass

    def _client_ip(self) -> str:
        return self.client_address[0] if self.client_address else "unknown"

    def _send_json(self, status_code: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        _log(f"📤 [HTTP] {self._client_ip()} <- {status_code} Response: {json.dumps(data)}")

    def _is_authorized(self) -> bool:
        expected_secret = self.server.secret
        if not expected_secret:
            return True

        # Check X-Webhook-Secret header
        header_secret = self.headers.get("X-Webhook-Secret") or self.headers.get("x-webhook-secret")
        if header_secret and header_secret.strip() == expected_secret.strip():
            return True

        # Check Authorization header (Bearer token)
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if token == expected_secret.strip():
                return True

        # Check URL query param ?secret=...
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if "secret" in query and query["secret"][0] == expected_secret:
            return True

        return False

    def do_GET(self):
        parsed = urlparse(self.path)
        _log(f"🔍 [HTTP] {self._client_ip()} -> GET {self.path}")
        if parsed.path in ("/", "/health", "/status"):
            uptime = round(time.time() - _start_time, 1)
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "immich-static-webhook",
                    "uptime_seconds": uptime,
                    "queue_size": self.server.worker_queue.qsize(),
                    "library_mount": str(self.server.folder) if self.server.folder else None,
                },
            )
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        _log(f"📩 [HTTP] {self._client_ip()} -> POST {self.path} ({content_length} bytes)")

        if parsed.path not in ("/webhook", "/webhook/"):
            self._send_json(404, {"error": f"Endpoint not found: {parsed.path}"})
            return

        if not self._is_authorized():
            _log("⚠️  [AUTH] Unauthorized webhook request rejected (invalid secret).")
            self._send_json(401, {"error": "Unauthorized. Provide valid secret token."})
            return

        # Parse request body
        if content_length <= 0:
            self._send_json(400, {"error": "Empty request body"})
            return

        try:
            raw_body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(raw_body)
        except Exception as e:
            self._send_json(400, {"error": f"Malformed JSON: {e}"})
            return

        asset_id, asset_data = extract_asset_info_from_payload(payload)
        if not asset_id:
            _log(f"⚠️  [WEBHOOK] Received webhook with no identifiable asset ID: {payload}")
            self._send_json(400, {"error": "No asset ID found in webhook payload"})
            return

        # Quick pre-filter: if payload explicitly declares non-video asset
        asset_type = asset_data.get("type")
        if asset_type and asset_type.upper() not in ("VIDEO", "LIVE_PHOTO_VIDEO"):
            _log(f"ℹ️  [FILTER] Skipped non-video asset {asset_id} (type: {asset_type})")
            self._send_json(200, {"status": "ignored", "reason": f"Asset type is {asset_type}, not VIDEO"})
            return

        # Enqueue for asynchronous background processing to prevent webhook timeout
        self.server.worker_queue.put((asset_id, asset_data))
        qsize = self.server.worker_queue.qsize()
        _log(f"📥 [QUEUE] Asset {asset_id} enqueued successfully (queue depth: {qsize})")

        self._send_json(202, {"status": "queued", "asset_id": asset_id, "queue_depth": qsize})


def _worker_loop(server: WebhookServer, worker_id: int):
    """Background consumer thread that processes video assets from the queue."""
    while not _server_stop_event.is_set():
        try:
            asset_id, asset_data = server.worker_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            _process_queued_asset(server, asset_id, asset_data)
        except Exception as e:
            print(f"❌ [WORKER-{worker_id}] Error processing asset {asset_id}: {e}")
        finally:
            server.worker_queue.task_done()


def _process_queued_asset(server: WebhookServer, asset_id: str, asset_data: Dict[str, Any]):
    """Analyzes a single video asset and syncs classification to Immich."""
    client = server.client
    ckpt = server.ckpt
    thresholds = server.thresholds

    # 1. Fetch full metadata if necessary
    full_info = dict(asset_data)
    if not full_info.get("originalPath") or not full_info.get("type"):
        try:
            fetched = client.get_asset_info(asset_id)
            if fetched:
                full_info.update(fetched)
        except Exception as e:
            print(f"⚠️  [WEBHOOK] Failed to fetch asset info for {asset_id}: {e}")

    # Check asset type
    asset_type = (full_info.get("type") or "").upper()
    if asset_type and asset_type not in ("VIDEO", "LIVE_PHOTO_VIDEO"):
        _log(f"ℹ️  [SKIP] Asset {asset_id}: type is {asset_type} (not VIDEO)")
        return

    # Check already processed in checkpoint
    if ckpt.is_done(asset_id):
        existing = ckpt.get(asset_id) or {}
        _log(f"ℹ️  [SKIP] Asset {asset_id} already classified as '{existing.get('decision', '?')}'. Skipping.")
        return

    # 2. Resolve local filesystem path
    video_path = resolve_asset_file_path(
        full_info,
        folder=server.folder,
        remote_prefix=server.remote_prefix,
        local_prefix=server.local_prefix,
    )

    temp_downloaded_file: Optional[Path] = None

    if video_path and video_path.is_file():
        pass
    else:
        # Fallback: Download asset stream via Immich API
        orig_name = full_info.get("originalFileName") or f"{asset_id}.mp4"
        ext = Path(orig_name).suffix.lower() or ".mp4"
        if ext not in VIDEO_EXTENSIONS:
            ext = ".mp4"

        _log(f"📡 [DOWNLOAD] Local file not found on mount. Streaming asset {asset_id} via Immich API...")
        temp_dir = Path(tempfile.gettempdir()) / "immich_static_cache"
        temp_file = temp_dir / f"stream_{asset_id}{ext}"
        try:
            client.download_asset(asset_id, temp_file)
            video_path = temp_file
            temp_downloaded_file = temp_file
        except Exception as e:
            _log(f"❌ [ERROR] Failed to download asset {asset_id}: {e}")
            return

    # 3. Run detection engine
    orig_name = full_info.get("originalFileName") or video_path.name
    _log(f"🎬 [DETECT] Analyzing motion for '{orig_name}' (ID: {asset_id})...")

    try:
        row = detect_video(video_path, thresholds)
    except Exception as e:
        row = {f: "" for f in LOG_FIELDS}
        row["decision"] = f"error: {e}"

    row["asset_id"] = asset_id
    row["original_file_name"] = orig_name
    row["filename"] = str(video_path.resolve())

    # Record in checkpoint under both asset_id and filepath for seamless lookup
    ckpt.record(asset_id, row)
    ckpt.record(str(video_path.resolve()), row)

    decision = row.get("decision", "unknown")
    motion = row.get("global_motion_score", "?")
    zones = row.get("active_zone_ratio", "?")
    duration = row.get("duration_s", "?")
    _log(f"📊 [DECISION] '{orig_name}' -> {decision.upper()} | motion: {motion} | active_zones: {zones} | duration: {duration}s")

    # 4. Apply targeted tag & album sync to Immich
    if decision in ("static", "review", "dynamic"):
        sync_res = sync_single_asset(
            client=client,
            asset_id=asset_id,
            decision=decision,
            sync_tags=True,
            sync_albums=True,
            include_dynamic=server.include_dynamic,
            dry_run=server.dry_run,
        )
        _log(f"🏷️  [SYNC] Immich updated for '{orig_name}': Tagged={sync_res.get('tagged')} | AlbumAdded={sync_res.get('album_added')}")

    # 5. Optional sharpest frame extraction
    if server.extract_frame and decision == "static" and video_path.is_file():
        out_dir = server.extract_dir or Path("./extracted_frames")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(orig_name).stem
        out_frame = out_dir / f"{stem}.jpg"
        try:
            extract_one_frame(video_path, out_frame, fmt="jpg", quality=95)
            _log(f"📸 [EXTRACT] Extracted frame: {out_frame.name}")
            if not server.dry_run:
                client.upload_asset(out_frame)
                _log(f"⬆️  [UPLOAD] Uploaded extracted frame to Immich")
        except Exception as e:
            _log(f"⚠️  [EXTRACT] Frame extraction failed: {e}")

    # 6. Cleanup temporary download
    if temp_downloaded_file and temp_downloaded_file.exists():
        try:
            temp_downloaded_file.unlink()
        except OSError:
            pass


def _signal_handler(sig, frame):
    global _server_stop_event
    print("\n🛑 Shutting down webhook server gracefully...")
    _server_stop_event.set()


def run_serve(args: Any):
    """Starts the real-time Immich webhook server daemon."""
    global _server_stop_event
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    host = getattr(args, "host", None) or os.environ.get("WEBHOOK_HOST", "0.0.0.0")
    port = int(getattr(args, "port", None) or os.environ.get("WEBHOOK_PORT", 8080))
    secret = getattr(args, "secret", None) or os.environ.get("WEBHOOK_SECRET")

    folder_str = getattr(args, "folder", None) or os.environ.get("IMMICH_LIBRARY_PATH")
    folder = Path(folder_str).resolve() if folder_str else None

    remote_prefix = getattr(args, "remote_prefix", None) or os.environ.get("IMMICH_REMOTE_PATH_PREFIX")
    local_prefix = getattr(args, "local_prefix", None) or os.environ.get("IMMICH_LOCAL_PATH_PREFIX")

    db_path_str = getattr(args, "db_path", None) or os.environ.get("DB_PATH", "./" + CHECKPOINT_FILENAME)
    ckpt = Checkpoint(Path(db_path_str).resolve())

    sensitivity = getattr(args, "sensitivity", "medium") or "medium"
    thresholds = SENSITIVITY_PRESETS.get(sensitivity, SENSITIVITY_PRESETS["medium"])

    workers_count = min(int(getattr(args, "workers", None) or 2), ENV["max_workers"])
    include_dynamic = getattr(args, "include_dynamic", True)
    extract_frame = bool(getattr(args, "extract", False))
    extract_dir = Path(getattr(args, "extract_dir", "./extracted_frames")).resolve()
    dry_run = bool(getattr(args, "dry_run", False))

    api_url = getattr(args, "api_url", None) or os.environ.get("IMMICH_API_URL")
    api_key = getattr(args, "api_key", None) or os.environ.get("IMMICH_API_KEY")

    if not api_url or not api_key:
        print("❌ Error: Immich API URL and API Key are required for webhook mode.")
        print("   Set IMMICH_API_URL and IMMICH_API_KEY in .env or pass --api-url and --api-key")
        sys.exit(1)

    client = ImmichClient(api_url, api_key)
    try:
        ver = client.ping()
        print(f"🌐 Connected to Immich Server ({ver.get('major', '')}.{ver.get('minor', '')}.{ver.get('patch', '')})")
    except Exception as e:
        print(f"❌ Could not connect to Immich API: {e}")
        sys.exit(1)

    server = WebhookServer(
        (host, port),
        WebhookRequestHandler,
        client=client,
        ckpt=ckpt,
        thresholds=thresholds,
        folder=folder,
        remote_prefix=remote_prefix,
        local_prefix=local_prefix,
        secret=secret,
        include_dynamic=include_dynamic,
        extract_frame=extract_frame,
        extract_dir=extract_dir,
        dry_run=dry_run,
    )

    # Start background worker threads
    worker_threads = []
    for i in range(workers_count):
        t = threading.Thread(target=_worker_loop, args=(server, i + 1), daemon=True)
        t.start()
        worker_threads.append(t)

    print("\n" + "=" * 65)
    print("🚀 Immich Static Webhook Daemon running!")
    print(f"   • Listening on     : http://{host}:{port}/webhook")
    print(f"   • Health check     : http://{host}:{port}/health")
    print(f"   • Authentication   : {'🔒 Secret enforced' if secret else '⚠️  Public (no secret set)'}")
    print(f"   • Local mount      : {folder if folder else '📡 API stream / download fallback'}")
    if remote_prefix and local_prefix:
        print(f"   • Path remapping   : {remote_prefix} -> {local_prefix}")
    print(f"   • Sensitivity      : {sensitivity}")
    print(f"   • Worker threads   : {workers_count}")
    print(f"   • Dynamic videos   : {'Synced' if include_dynamic else 'Ignored'}")
    print("=" * 65 + "\n")
    print("Press Ctrl+C to stop.\n")

    # Run server loop with periodic interruption checks
    server.timeout = 1.0
    while not _server_stop_event.is_set():
        server.handle_request()

    print("🛑 Draining queued tasks...")
    server.worker_queue.join()
    server.server_close()
    print("👋 Webhook server stopped cleanly.")
