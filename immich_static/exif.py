"""
immich_static.exif — EXIF, GPS, and Adobe XMP preservation engine for extracted frames.
Constructs standard EXIF IFDs (0th IFD, Exif IFD, GPS IFD), Adobe XMP metadata packets,
PNG text chunks, and synchronizes filesystem timestamps.
"""

import datetime
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import Base, IFD, GPS
from PIL.PngImagePlugin import PngInfo

DEFAULT_IMAGE_TAGS = [os.environ.get("TAG_EXTRACTED", "video:extracted")]


@dataclass
class VideoMetadata:
    """Technical and container metadata probed from a video container/stream."""
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    aspect_ratio: str = "unknown"
    has_audio: bool = False
    codec: str = "unknown"
    filesize_bytes: int = 0
    creation_time: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)


def parse_iso6709_location(loc_str: Optional[str]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Parses ISO 6709 geographic location string into (latitude, longitude, altitude).
    Handles formats like:
      +37.7749-122.4194/
      +37.7749-122.4194+015.000/
      +28.5355+077.3910/
      -33.8688+151.2093-5.5/
    """
    if not loc_str or not isinstance(loc_str, str):
        return None, None, None

    cleaned = loc_str.strip().rstrip("/")
    if not cleaned:
        return None, None, None

    # Matches (+/-lat)(+/-lon)(+/-alt optional)
    pattern = r"^([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?"
    match = re.match(pattern, cleaned)
    if not match:
        return None, None, None

    try:
        lat = float(match.group(1))
        lon = float(match.group(2))
        alt = float(match.group(3)) if match.group(3) is not None else None
        return lat, lon, alt
    except (ValueError, TypeError):
        return None, None, None


def parse_iso_datetime(dt_str: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[datetime.datetime]]:
    """
    Parses an ISO 8601 or common video creation timestamp into:
      (exif_formatted_str 'YYYY:MM:DD HH:MM:SS', tz_offset_str '+HH:MM', datetime_obj)
    Rejects any uninitialized / epoch-zero timestamps on or before year 1971.
    """
    if not dt_str or not isinstance(dt_str, str):
        return None, None, None

    cleaned = dt_str.strip()
    if not cleaned:
        return None, None, None

    # Try fromisoformat first (native in Python 3.11+ with timezone support)
    try:
        iso_clean = cleaned.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(iso_clean)
        if dt.year <= 1971:
            return None, None, None
        exif_str = dt.strftime("%Y:%m:%d %H:%M:%S")
        offset = None
        if dt.tzinfo is not None:
            offset_raw = dt.strftime("%z")
            if offset_raw and len(offset_raw) == 5:
                offset = f"{offset_raw[:3]}:{offset_raw[3:]}"
        return exif_str, offset, dt
    except Exception:
        pass

    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(cleaned, fmt)
            if fmt.endswith("Z"):
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            if dt.year <= 1971:
                return None, None, None
            exif_str = dt.strftime("%Y:%m:%d %H:%M:%S")
            offset = None
            if dt.tzinfo is not None:
                offset_raw = dt.strftime("%z")
                if offset_raw and len(offset_raw) == 5:
                    offset = f"{offset_raw[:3]}:{offset_raw[3:]}"
            return exif_str, offset, dt
        except ValueError:
            pass

    # Regex fallback for YYYY-MM-DD HH:MM:SS
    match = re.match(r"(\d{4})[-:.](\d{2})[-:.](\d{2})[T ](\d{2}):(\d{2}):(\d{2})", cleaned)
    if match:
        y, m, d, hh, mm, ss = match.groups()
        if int(y) <= 1971:
            return None, None, None
        exif_str = f"{y}:{m}:{d} {hh}:{mm}:{ss}"
        try:
            dt_obj = datetime.datetime(int(y), int(m), int(d), int(hh), int(mm), int(ss))
        except ValueError:
            dt_obj = None
        return exif_str, None, dt_obj

    return None, None, None


def parse_filename_datetime(filename: str) -> Optional[datetime.datetime]:
    """
    Infers creation datetime from filename heuristics:
    1. Instagram Story Snowflake IDs (story_<snowflake_id>):
       Decodes timestamp in ms: (snowflake_id >> 23) + 1314220021721.
    2. Date-time formats like YYYYMMDD_HHMMSS or YYYY-MM-DD-HHMMSS.
    3. Date formats like YYYY-MM-DD or YYYYMMDD.
    4. 10-digit UNIX epoch timestamps.
    """
    if not filename:
        return None
    name = Path(filename).stem

    # 1. Instagram Story Snowflake ID: story_<id>
    m_story = re.search(r"story_(\d{18,20})", name)
    if not m_story:
        m_story = re.search(r"(?:^|_)(\d{18,20})(?:_|$)", name)
    if m_story:
        try:
            snowflake_id = int(m_story.group(1))
            ts_ms = (snowflake_id >> 23) + 1314220021721
            dt = datetime.datetime.fromtimestamp(ts_ms / 1000.0, tz=datetime.timezone.utc)
            if 2011 <= dt.year <= 2035:
                return dt
        except Exception:
            pass

    # 2. YYYYMMDD_HHMMSS or YYYYMMDD-HHMMSS (e.g. 20260925_234636531, 20260519_182208)
    m_dt = re.search(r"(\d{4})(\d{2})(\d{2})[_\-T ](\d{2})(\d{2})(\d{2})", name)
    if m_dt:
        try:
            y, m, d, hh, mm, ss = map(int, m_dt.groups())
            if 1980 <= y <= 2035 and 1 <= m <= 12 and 1 <= d <= 31 and 0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59:
                return datetime.datetime(y, m, d, hh, mm, ss, tzinfo=datetime.timezone.utc)
        except Exception:
            pass

    # 3. YYYY-MM-DD-HHMM or YYYYMMDD-HHMM (e.g. 20230908-0001)
    m_date = re.search(r"(\d{4})[-_.]?(\d{2})[-_.]?(\d{2})[-_](\d{4})", name)
    if m_date:
        try:
            y, m, d = int(m_date.group(1)), int(m_date.group(2)), int(m_date.group(3))
            num = int(m_date.group(4))
            hh = min(23, num // 100)
            mm = min(59, num % 100)
            if 1980 <= y <= 2035 and 1 <= m <= 12 and 1 <= d <= 31:
                return datetime.datetime(y, m, d, hh, mm, 0, tzinfo=datetime.timezone.utc)
        except Exception:
            pass

    # 4. YYYY-MM-DD or YYYY_MM_DD
    m_donly = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", name)
    if m_donly:
        try:
            y, m, d = map(int, m_donly.groups())
            if 1980 <= y <= 2035 and 1 <= m <= 12 and 1 <= d <= 31:
                return datetime.datetime(y, m, d, 12, 0, 0, tzinfo=datetime.timezone.utc)
        except Exception:
            pass

    # 5. Unix timestamp in seconds (10 digits starting with 14-17)
    m_ts = re.search(r"(?:^|_)(\b1[4-7]\d{8}\b)(?:_|$)", name)
    if m_ts:
        try:
            ts = int(m_ts.group(1))
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            if 2014 <= dt.year <= 2035:
                return dt
        except Exception:
            pass

    return None


def parse_sidecar_metadata(video_path: Path) -> Dict[str, Any]:
    """
    Parses XMP or JSON (Google Takeout) sidecar files located next to video_path.
    Looks for video.ext.xmp, video.xmp, video.ext.json, video.json.
    """
    video_p = Path(video_path)
    candidates = [
        video_p.with_suffix(video_p.suffix + ".xmp"),
        video_p.with_suffix(".xmp"),
        video_p.with_suffix(video_p.suffix + ".json"),
        video_p.with_suffix(".json"),
    ]

    result: Dict[str, Any] = {}

    for cand in candidates:
        if not cand.exists() or not cand.is_file():
            continue

        ext = cand.suffix.lower()
        if ext == ".xmp":
            try:
                content = cand.read_text(encoding="utf-8", errors="ignore")
                # DateTimeOriginal / DateCreated / CreateDate / ModifyDate
                m_dt = (
                    re.search(r"<(?:exif:DateTimeOriginal|photoshop:DateCreated|xmp:CreateDate)>([^<]+)</", content)
                    or re.search(r"(?:exif:DateTimeOriginal|photoshop:DateCreated|xmp:CreateDate)=\"([^\"]+)\"", content)
                    or re.search(r"<(?:xmp:ModifyDate|dc:date)>([^<]+)</", content)
                )
                if m_dt and "creation_time" not in result:
                    raw_dt = m_dt.group(1).strip()
                    _, _, dt_obj = parse_iso_datetime(raw_dt)
                    if dt_obj and dt_obj.year > 1971:
                        result["creation_time"] = raw_dt

                # Camera Make & Model
                m_make = re.search(r"<tiff:Make>([^<]+)</", content) or re.search(r"tiff:Make=\"([^\"]+)\"", content)
                if m_make and "make" not in result:
                    result["make"] = m_make.group(1).strip()

                m_model = re.search(r"<tiff:Model>([^<]+)</", content) or re.search(r"tiff:Model=\"([^\"]+)\"", content)
                if m_model and "model" not in result:
                    result["model"] = m_model.group(1).strip()

                # GPS Coordinates
                m_lat = re.search(r"<(?:exif:GPSLatitude)>([^<]+)</", content)
                m_lon = re.search(r"<(?:exif:GPSLongitude)>([^<]+)</", content)
                m_alt = re.search(r"<(?:exif:GPSAltitude)>([^<]+)</", content)
                if m_lat and m_lon and "latitude" not in result:
                    try:
                        result["latitude"] = float(m_lat.group(1).strip())
                        result["longitude"] = float(m_lon.group(1).strip())
                        result["altitude"] = float(m_alt.group(1).strip()) if m_alt else None
                    except ValueError:
                        pass

                # Tags / Keywords from digiKam:TagsList or dc:subject
                tags = []
                for m_tag in re.finditer(r"<rdf:li>([^<]+)</rdf:li>", content):
                    t = m_tag.group(1).strip()
                    if t and t not in tags:
                        tags.append(t)
                if tags:
                    result.setdefault("tags", []).extend(tags)
            except Exception:
                pass

        elif ext == ".json":
            try:
                import json
                data = json.loads(cand.read_text(encoding="utf-8", errors="ignore"))
                taken = (
                    data.get("photoTakenTime", {}).get("timestamp")
                    or data.get("creationTime", {}).get("timestamp")
                )
                if taken and "creation_time" not in result:
                    ts = int(taken)
                    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
                    if dt.year > 1971:
                        result["creation_time"] = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

                geo = data.get("geoData", {})
                if (geo.get("latitude") or geo.get("longitude")) and "latitude" not in result:
                    result["latitude"] = float(geo.get("latitude", 0.0))
                    result["longitude"] = float(geo.get("longitude", 0.0))
                    result["altitude"] = float(geo.get("altitude", 0.0))
            except Exception:
                pass

    return result


def deg_to_dms(deg_float: float) -> Tuple[float, float, float]:
    """Converts decimal degrees to EXIF GPS tuple (degrees, minutes, seconds)."""
    deg = abs(deg_float)
    d = int(deg)
    m_float = (deg - d) * 60.0
    m = int(m_float)
    s = round((m_float - m) * 60.0, 4)
    if s >= 60.0:
        s = 0.0
        m += 1
    if m >= 60:
        m = 0
        d += 1
    return (float(d), float(m), s)


def probe_video(
    video_path: Path,
    original_file_name: Optional[str] = None,
    date_hint: Optional[str] = None,
    metadata_hint: Optional[Dict[str, Any]] = None,
    timeout: int = 15,
) -> VideoMetadata:
    """
    Probe video metadata using ffprobe and sidecars (.xmp / .json).
    Returns populated VideoMetadata object including format/stream tags,
    creation time, camera make/model, and GPS location.
    Guarantees epoch-zero (<= 1971) dummy dates are discarded and replaced
    by sidecar, Immich date_hint, or filename heuristics.
    """
    video_path = Path(video_path)
    meta = VideoMetadata()
    if not video_path.exists() or not video_path.is_file():
        return meta

    try:
        meta.filesize_bytes = video_path.stat().st_size
    except Exception:
        meta.filesize_bytes = 0

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_format",
        "-show_streams",
        "-of", "json",
        str(video_path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            return meta

        data = json.loads(proc.stdout or "{}")

        # Format tags
        fmt = data.get("format", {})
        fmt_tags = fmt.get("tags", {})
        all_tags: dict = {}
        if isinstance(fmt_tags, dict):
            all_tags.update(fmt_tags)

        # Duration
        if "duration" in fmt:
            try:
                meta.duration_s = float(fmt["duration"])
            except (ValueError, TypeError):
                pass

        # Stream tags & video/audio attributes
        streams = data.get("streams", [])
        for s in streams:
            stype = s.get("codec_type")
            s_tags = s.get("tags", {})
            if isinstance(s_tags, dict):
                for k, v in s_tags.items():
                    if k not in all_tags:
                        all_tags[k] = v

            if stype == "video" and meta.width == 0:
                meta.width = int(s.get("width", 0))
                meta.height = int(s.get("height", 0))
                meta.codec = s.get("codec_name", "unknown")
                r_rate = s.get("r_frame_rate", "0/1")
                if "/" in r_rate:
                    num, den = r_rate.split("/")
                    meta.fps = float(num) / float(den) if float(den) != 0 else 0.0
                else:
                    meta.fps = float(r_rate) if r_rate else 0.0

                if meta.width > 0 and meta.height > 0:
                    ratio = meta.width / meta.height
                    if 0.50 <= ratio <= 0.60:
                        meta.aspect_ratio = "9:16"
                    elif 0.95 <= ratio <= 1.05:
                        meta.aspect_ratio = "1:1"
                    elif 0.75 <= ratio <= 0.85:
                        meta.aspect_ratio = "4:5"
                    elif 1.70 <= ratio <= 1.85:
                        meta.aspect_ratio = "16:9"
                    else:
                        meta.aspect_ratio = f"{meta.width}:{meta.height}"
            elif stype == "audio":
                meta.has_audio = True

        meta.tags = all_tags

        # ffprobe creation time (validate > 1971)
        raw_creation = (
            all_tags.get("creation_time")
            or all_tags.get("com.apple.quicktime.creationdate")
            or all_tags.get("date")
        )
        if raw_creation:
            _, _, parsed_dt = parse_iso_datetime(raw_creation)
            if parsed_dt and parsed_dt.year > 1971:
                meta.creation_time = raw_creation
            else:
                meta.creation_time = None

        # Make & model
        meta.make = (
            all_tags.get("make")
            or all_tags.get("com.apple.quicktime.make")
            or all_tags.get("camera_make")
            or all_tags.get("android.manufacturer")
        )
        meta.model = (
            all_tags.get("model")
            or all_tags.get("com.apple.quicktime.model")
            or all_tags.get("camera_model")
            or all_tags.get("android.model")
        )

        # GPS location
        loc_str = (
            all_tags.get("location")
            or all_tags.get("location-eng")
            or all_tags.get("com.apple.quicktime.location.ISO6709")
        )
        meta.location = loc_str
        if loc_str:
            lat, lon, alt = parse_iso6709_location(loc_str)
            meta.latitude = lat
            meta.longitude = lon
            meta.altitude = alt

    except Exception:
        pass

    # ── Sidecar Metadata Enrichment (.xmp / .json) ───────────────
    sidecar = parse_sidecar_metadata(video_path)
    if sidecar.get("creation_time") and not meta.creation_time:
        meta.creation_time = sidecar["creation_time"]
    if sidecar.get("make") and not meta.make:
        meta.make = sidecar["make"]
    if sidecar.get("model") and not meta.model:
        meta.model = sidecar["model"]
    if sidecar.get("latitude") is not None and meta.latitude is None:
        meta.latitude = sidecar["latitude"]
        meta.longitude = sidecar["longitude"]
        meta.altitude = sidecar.get("altitude")
    if sidecar.get("tags"):
        existing_kw = meta.tags.get("keywords", "")
        kw_list = [k.strip() for k in existing_kw.split(",") if k.strip()] if existing_kw else []
        for t in sidecar["tags"]:
            if t not in kw_list:
                kw_list.append(t)
        meta.tags["keywords"] = ", ".join(kw_list)

    # ── Date Hint Fallback (Immich API / SQLite Checkpoint) ───────
    if not meta.creation_time and date_hint:
        _, _, hint_dt = parse_iso_datetime(str(date_hint))
        if hint_dt and hint_dt.year > 1971:
            meta.creation_time = str(date_hint)

    # ── Filename Heuristics Fallback (Instagram Story, ISO dates) ─
    if not meta.creation_time:
        fn_candidate = original_file_name or video_path.name
        fn_dt = parse_filename_datetime(fn_candidate)
        if fn_dt and fn_dt.year > 1971:
            meta.creation_time = fn_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Filesystem mtime Fallback ────────────────────────────────
    if not meta.creation_time:
        try:
            mtime = video_path.stat().st_mtime
            if mtime > 31536000:  # > 1971-01-01
                m_dt = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
                if m_dt.year > 1971:
                    meta.creation_time = m_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass

    # ── Metadata Hint (GPS, Make, Model) ─────────────────────────
    if metadata_hint:
        if meta.latitude is None and metadata_hint.get("latitude") is not None:
            try:
                meta.latitude = float(metadata_hint["latitude"])
                meta.longitude = float(metadata_hint.get("longitude", 0.0))
                if metadata_hint.get("altitude") is not None:
                    meta.altitude = float(metadata_hint["altitude"])
            except (ValueError, TypeError):
                pass
        if not meta.make and metadata_hint.get("make"):
            meta.make = str(metadata_hint["make"])
        if not meta.model and metadata_hint.get("model"):
            meta.model = str(metadata_hint["model"])

    return meta


def build_image_exif(
    meta: VideoMetadata,
    video_path: Optional[Path] = None,
    tags: Optional[List[str]] = None,
) -> Image.Exif:
    """
    Constructs a Pillow Exif object populated with camera make/model,
    creation datetime, GPS coordinates, orientation, software, description,
    and keyword tags (XPKeywords, UserComment).
    """
    exif = Image.Exif()

    # Orientation: 1 (Normal / Top-Left) since extracted frame is already decoded right-side up
    exif[Base.Orientation] = 1

    # Camera Make & Model
    meta_tags = meta.tags or {}
    make = (
        meta.make
        or meta_tags.get("make")
        or meta_tags.get("com.apple.quicktime.make")
        or meta_tags.get("camera_make")
        or meta_tags.get("android.manufacturer")
    )
    if make:
        exif[Base.Make] = str(make).strip()

    model = (
        meta.model
        or meta_tags.get("model")
        or meta_tags.get("com.apple.quicktime.model")
        or meta_tags.get("camera_model")
        or meta_tags.get("android.model")
    )
    if model:
        exif[Base.Model] = str(model).strip()

    # Software / Encoder
    software = (
        meta_tags.get("software")
        or meta_tags.get("encoder")
        or meta_tags.get("com.apple.quicktime.software")
        or "ImmichStatic"
    )
    if software:
        exif[Base.Software] = str(software).strip()

    keyword_list: List[str] = list(tags if tags is not None else DEFAULT_IMAGE_TAGS)
    if "keywords" in meta_tags:
        for kw in str(meta_tags["keywords"]).split(","):
            cleaned_kw = kw.strip()
            if cleaned_kw and cleaned_kw not in keyword_list:
                keyword_list.append(cleaned_kw)

    # Description & Copyright
    description = (
        meta_tags.get("description")
        or meta_tags.get("title")
        or meta_tags.get("comment")
        or meta_tags.get("synopsis")
    )
    if description:
        exif[Base.ImageDescription] = str(description).strip()
    elif keyword_list:
        exif[Base.ImageDescription] = ", ".join(keyword_list)

    copyright_info = meta_tags.get("copyright") or meta_tags.get("artist") or meta_tags.get("author")
    if copyright_info:
        exif[Base.Copyright] = str(copyright_info).strip()

    # Windows XP Keyword tags
    if keyword_list:
        kw_str = "; ".join(keyword_list)
        exif[0x9C9E] = kw_str.encode("utf-16le") + b"\x00\x00"
        exif[0x9C9B] = kw_str.encode("utf-16le") + b"\x00\x00"

    if description:
        exif[0x9C9C] = str(description).encode("utf-16le") + b"\x00\x00"
    elif keyword_list:
        exif[0x9C9C] = (", ".join(keyword_list)).encode("utf-16le") + b"\x00\x00"

    # Creation Time
    creation_raw = (
        meta.creation_time
        or meta_tags.get("creation_time")
        or meta_tags.get("com.apple.quicktime.creationdate")
        or meta_tags.get("date")
    )
    exif_dt, tz_offset, dt_obj = parse_iso_datetime(creation_raw)

    exif_ifd = exif.get_ifd(IFD.Exif)
    if exif_dt:
        exif[Base.DateTime] = exif_dt
        exif_ifd[Base.DateTimeOriginal] = exif_dt
        exif_ifd[Base.DateTimeDigitized] = exif_dt
        if tz_offset:
            exif_ifd[Base.OffsetTime] = tz_offset
            exif_ifd[Base.OffsetTimeOriginal] = tz_offset
            exif_ifd[Base.OffsetTimeDigitized] = tz_offset

    # UserComment in Exif IFD
    user_comment_text = description or (", ".join(keyword_list) if keyword_list else "ImmichStatic")
    exif_ifd[Base.UserComment] = b"ASCII\x00\x00\x00" + str(user_comment_text).encode("ascii", "replace")

    # GPS Info
    lat = meta.latitude
    lon = meta.longitude
    alt = meta.altitude

    if lat is None or lon is None:
        loc_str = meta.location or meta_tags.get("location") or meta_tags.get("location-eng") or meta_tags.get("com.apple.quicktime.location.ISO6709")
        if loc_str:
            p_lat, p_lon, p_alt = parse_iso6709_location(loc_str)
            if p_lat is not None and p_lon is not None:
                lat, lon = p_lat, p_lon
                if alt is None and p_alt is not None:
                    alt = p_alt

    if lat is not None and lon is not None:
        gps_ifd = exif.get_ifd(IFD.GPSInfo)
        gps_ifd[GPS.GPSLatitudeRef] = "N" if lat >= 0 else "S"
        gps_ifd[GPS.GPSLatitude] = deg_to_dms(lat)
        gps_ifd[GPS.GPSLongitudeRef] = "E" if lon >= 0 else "W"
        gps_ifd[GPS.GPSLongitude] = deg_to_dms(lon)

        if alt is not None:
            gps_ifd[GPS.GPSAltitudeRef] = b"\x00" if alt >= 0 else b"\x01"
            gps_ifd[GPS.GPSAltitude] = float(abs(alt))

        if dt_obj is not None:
            gps_ifd[GPS.GPSDateStamp] = dt_obj.strftime("%Y:%m:%d")
            gps_ifd[GPS.GPSTimeStamp] = (
                float(dt_obj.hour),
                float(dt_obj.minute),
                float(dt_obj.second),
            )

    return exif


def build_png_info(
    meta: VideoMetadata,
    video_path: Optional[Path] = None,
    tags: Optional[List[str]] = None,
) -> PngInfo:
    """Constructs PNG textual metadata chunks including keyword tags."""
    png_info = PngInfo()
    meta_tags = meta.tags or {}

    creation_raw = (
        meta.creation_time
        or meta_tags.get("creation_time")
        or meta_tags.get("com.apple.quicktime.creationdate")
        or meta_tags.get("date")
    )
    if creation_raw:
        png_info.add_text("Creation Time", str(creation_raw))

    make = meta.make or meta_tags.get("make") or meta_tags.get("com.apple.quicktime.make")
    if make:
        png_info.add_text("Make", str(make))

    model = meta.model or meta_tags.get("model") or meta_tags.get("com.apple.quicktime.model")
    if model:
        png_info.add_text("Model", str(model))

    software = meta_tags.get("software") or meta_tags.get("encoder") or "ImmichStatic"
    if software:
        png_info.add_text("Software", str(software))

    desc = meta_tags.get("description") or meta_tags.get("title") or meta_tags.get("comment")
    if desc:
        png_info.add_text("Description", str(desc))

    keyword_list: List[str] = list(tags if tags is not None else DEFAULT_IMAGE_TAGS)
    if keyword_list:
        png_info.add_text("Keywords", ", ".join(keyword_list))
        png_info.add_text("Comment", ", ".join(keyword_list))

    return png_info


def build_xmp_packet(meta: VideoMetadata, tags: Optional[List[str]] = None) -> bytes:
    """Constructs an Adobe XMP packet with Dublin Core, Exif, and Photoshop metadata for Immich / Lightroom."""
    keyword_list: List[str] = list(tags if tags is not None else DEFAULT_IMAGE_TAGS)
    meta_tags = meta.tags or {}
    if "keywords" in meta_tags:
        for kw in str(meta_tags["keywords"]).split(","):
            cleaned_kw = kw.strip()
            if cleaned_kw and cleaned_kw not in keyword_list:
                keyword_list.append(cleaned_kw)

    desc_text = (
        meta_tags.get("description")
        or meta_tags.get("title")
        or (", ".join(keyword_list) if keyword_list else "")
    )

    def _escape(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    # Date block for XMP
    date_block = ""
    creation_raw = (
        meta.creation_time
        or meta_tags.get("creation_time")
        or meta_tags.get("com.apple.quicktime.creationdate")
        or meta_tags.get("date")
    )
    _, tz_off, dt_obj = parse_iso_datetime(creation_raw)
    if dt_obj is not None and dt_obj.year > 1971:
        if dt_obj.tzinfo is not None:
            iso_dt = dt_obj.isoformat()
        elif tz_off:
            iso_dt = f"{dt_obj.strftime('%Y-%m-%dT%H:%M:%S')}{tz_off}"
        else:
            iso_dt = dt_obj.strftime("%Y-%m-%dT%H:%M:%SZ")

        date_block = f"""
  <rdf:Description rdf:about=""
    xmlns:exif="http://ns.adobe.com/exif/1.0/"
    xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"
    xmlns:xmp="http://ns.adobe.com/xap/1.0/">
   <exif:DateTimeOriginal>{_escape(iso_dt)}</exif:DateTimeOriginal>
   <photoshop:DateCreated>{_escape(iso_dt)}</photoshop:DateCreated>
   <xmp:CreateDate>{_escape(iso_dt)}</xmp:CreateDate>
  </rdf:Description>"""

    bag_items = "".join(f"<rdf:li>{_escape(t)}</rdf:li>" for t in keyword_list)
    xmp_str = f"""<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:lr="http://lightroom.adobe.com/xap/1.0/">
   <dc:description>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">{_escape(str(desc_text))}</rdf:li>
    </rdf:Alt>
   </dc:description>
   <dc:subject>
    <rdf:Bag>
     {bag_items}
    </rdf:Bag>
   </dc:subject>
   <lr:hierarchicalSubject>
    <rdf:Bag>
     {bag_items}
    </rdf:Bag>
   </lr:hierarchicalSubject>
  </rdf:Description>{date_block}
 </rdf:RDF>
</x:xmpmeta>"""
    return xmp_str.encode("utf-8")


def run_exiftool_copy(source_video: Path, dest_image: Path) -> bool:
    """
    Supplementary pass using exiftool if available on the system to copy
    deep vendor-specific maker notes. Returns True if successful.
    """
    if shutil.which("exiftool") is None:
        return False

    try:
        cmd = [
            "exiftool",
            "-tagsFromFile",
            str(source_video),
            "-all:all>all:all",
            "-wm",
            "c",
            "-Orientation=1",
            "-overwrite_original",
            str(dest_image),
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=10, check=False)
        return proc.returncode == 0
    except Exception:
        return False


def save_image_with_metadata(
    frame_bgr: np.ndarray,
    output_path: Path,
    source_video_path: Path,
    meta: VideoMetadata,
    fmt: str = "jpg",
    quality: int = 95,
    tags: Optional[List[str]] = None,
) -> bool:
    """
    Saves extracted OpenCV frame (BGR numpy array) to output_path with:
    1. EXIF metadata (Make, Model, DateTimeOriginal, GPSInfo, Orientation, Software, XPKeywords, ImageDescription)
    2. Adobe XMP packet (dc:subject, dc:description, lr:hierarchicalSubject, exif:DateTimeOriginal) for Immich / Lightroom
    3. PNG textual metadata chunks if output format is PNG
    4. Filesystem timestamps (os.utime) synced to source video or inferred creation datetime
    5. Optional exiftool supplementary copy if installed
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert BGR (OpenCV) to RGB (Pillow)
    if len(frame_bgr.shape) == 3:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame_bgr

    pil_img = Image.fromarray(frame_rgb)
    exif = build_image_exif(meta, source_video_path, tags=tags)
    xmp_bytes = build_xmp_packet(meta, tags=tags)

    fmt_lower = fmt.lower()
    if fmt_lower in ("jpg", "jpeg"):
        pil_img.save(
            str(output_path),
            format="JPEG",
            quality=int(quality),
            exif=exif,
            xmp=xmp_bytes,
            subsampling=0,
        )
    elif fmt_lower == "png":
        png_info = build_png_info(meta, source_video_path, tags=tags)
        pil_img.save(
            str(output_path),
            format="PNG",
            exif=exif,
            xmp=xmp_bytes,
            pnginfo=png_info,
            compress_level=3,
        )
    elif fmt_lower == "webp":
        pil_img.save(
            str(output_path),
            format="WEBP",
            quality=int(quality),
            exif=exif,
        )
    else:
        pil_img.save(str(output_path), exif=exif)

    # Supplementary pass with exiftool if available
    run_exiftool_copy(source_video_path, output_path)

    # Synchronize filesystem timestamps (mtime and atime)
    try:
        src_stat = source_video_path.stat()
        creation_raw = (
            meta.creation_time
            or (meta.tags or {}).get("creation_time")
            or (meta.tags or {}).get("com.apple.quicktime.creationdate")
        )
        _, _, dt_obj = parse_iso_datetime(creation_raw)
        if dt_obj is not None and dt_obj.year > 1971:
            ts = dt_obj.timestamp()
            os.utime(output_path, (src_stat.st_atime, ts))
        elif src_stat.st_mtime > 31536000:
            os.utime(output_path, (src_stat.st_atime, src_stat.st_mtime))
        else:
            now_ts = time.time()
            os.utime(output_path, (now_ts, now_ts))
    except Exception:
        pass

    return output_path.exists() and output_path.stat().st_size > 0
