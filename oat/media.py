"""
Media analysis utilities for OpenAgentTrace.
Extracts metrics from images, audio, and video files.
"""

import base64
import io
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib


@dataclass
class MediaMetadata:
    """Metadata extracted from media inputs/outputs."""
    media_type: str  # "image", "audio", "video", "document"
    format: str      # "png", "jpg", "mp3", "mp4", "pdf"
    size_bytes: int = 0
    
    # Image-specific
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    
    # Audio/Video-specific  
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    bit_rate: Optional[int] = None
    fps: Optional[float] = None
    
    # Token estimation (for vision/audio models)
    estimated_tokens: int = 0
    
    # Reference
    content_hash: Optional[str] = None
    filename: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def _get_bytes(data: Union[bytes, str, Path]) -> tuple:
    """Convert input to bytes and determine source type."""
    if isinstance(data, bytes):
        return data, None, None
    
    if isinstance(data, Path) or (isinstance(data, str) and os.path.exists(data)):
        path = Path(data)
        with open(path, 'rb') as f:
            return f.read(), path.name, path.suffix.lower().lstrip('.')
    
    # Check if base64 encoded
    if isinstance(data, str):
        # Handle data URLs: data:image/png;base64,iVBORw0...
        if data.startswith('data:'):
            header, b64_data = data.split(',', 1)
            mime_type = header.split(':')[1].split(';')[0]
            ext = mime_type.split('/')[-1]
            return base64.b64decode(b64_data), None, ext
        
        # Try raw base64
        try:
            return base64.b64decode(data), None, None
        except:
            pass
    
    return None, None, None


def _compute_hash(data: bytes) -> str:
    """Compute content hash for deduplication."""
    return hashlib.sha256(data).hexdigest()[:16]


def estimate_image_tokens(width: int, height: int, detail: str = "auto") -> int:
    """
    Estimate tokens for OpenAI vision models.
    Based on OpenAI's token calculation:
    - low detail: 85 tokens
    - high detail: 85 + 170 * tiles (where tile = 512x512)
    """
    if detail == "low":
        return 85
    
    # High detail calculation
    # Scale to fit in 2048x2048
    max_dim = max(width, height)
    if max_dim > 2048:
        scale = 2048 / max_dim
        width = int(width * scale)
        height = int(height * scale)
    
    # Scale shortest side to 768
    min_dim = min(width, height)
    if min_dim > 768:
        scale = 768 / min_dim
        width = int(width * scale)
        height = int(height * scale)
    
    # Count 512x512 tiles
    tiles_x = (width + 511) // 512
    tiles_y = (height + 511) // 512
    total_tiles = tiles_x * tiles_y
    
    return 85 + (170 * total_tiles)


def analyze_image(data: Union[bytes, str, Path]) -> Optional[MediaMetadata]:
    """
    Analyze an image and extract metadata.
    
    Args:
        data: Image bytes, file path, or base64 string
        
    Returns:
        MediaMetadata with dimensions, format, size, and estimated tokens
    """
    raw_bytes, filename, ext = _get_bytes(data)
    if not raw_bytes:
        return None
    
    try:
        from PIL import Image
        
        img = Image.open(io.BytesIO(raw_bytes))
        width, height = img.size
        
        # Determine format
        fmt = ext or (img.format.lower() if img.format else "unknown")
        
        # Channels
        mode_channels = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4, "CMYK": 4}
        channels = mode_channels.get(img.mode, 3)
        
        return MediaMetadata(
            media_type="image",
            format=fmt,
            size_bytes=len(raw_bytes),
            width=width,
            height=height,
            channels=channels,
            estimated_tokens=estimate_image_tokens(width, height),
            content_hash=_compute_hash(raw_bytes),
            filename=filename,
        )
    except ImportError:
        # PIL not installed - return basic metadata
        return MediaMetadata(
            media_type="image",
            format=ext or "unknown",
            size_bytes=len(raw_bytes),
            content_hash=_compute_hash(raw_bytes),
            filename=filename,
        )
    except Exception as e:
        return MediaMetadata(
            media_type="image",
            format=ext or "unknown", 
            size_bytes=len(raw_bytes) if raw_bytes else 0,
            content_hash=_compute_hash(raw_bytes) if raw_bytes else None,
        )


def analyze_audio(data: Union[bytes, str, Path]) -> Optional[MediaMetadata]:
    """
    Analyze an audio file and extract metadata.
    
    Args:
        data: Audio bytes, file path, or base64 string
        
    Returns:
        MediaMetadata with duration, sample rate, format, size
    """
    raw_bytes, filename, ext = _get_bytes(data)
    if not raw_bytes:
        return None
    
    metadata = MediaMetadata(
        media_type="audio",
        format=ext or "unknown",
        size_bytes=len(raw_bytes),
        content_hash=_compute_hash(raw_bytes),
        filename=filename,
    )
    
    try:
        from mutagen import File as MutagenFile
        
        # Write to temp file for mutagen
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=f'.{ext or "mp3"}', delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        
        try:
            audio = MutagenFile(tmp_path)
            if audio:
                metadata.duration_seconds = audio.info.length if hasattr(audio.info, 'length') else None
                metadata.sample_rate = audio.info.sample_rate if hasattr(audio.info, 'sample_rate') else None
                metadata.bit_rate = audio.info.bitrate if hasattr(audio.info, 'bitrate') else None
                
                # Estimate tokens for speech-to-text (rough: 1 token per 0.02 seconds)
                if metadata.duration_seconds:
                    metadata.estimated_tokens = int(metadata.duration_seconds / 0.02)
        finally:
            os.unlink(tmp_path)
            
    except ImportError:
        pass  # mutagen not installed
    except Exception:
        pass
    
    return metadata


def analyze_video(data: Union[bytes, str, Path]) -> Optional[MediaMetadata]:
    """
    Analyze a video file and extract metadata.
    
    Args:
        data: Video bytes, file path, or base64 string
        
    Returns:
        MediaMetadata with dimensions, duration, fps, format, size
    """
    raw_bytes, filename, ext = _get_bytes(data)
    if not raw_bytes:
        return None
    
    metadata = MediaMetadata(
        media_type="video",
        format=ext or "unknown",
        size_bytes=len(raw_bytes),
        content_hash=_compute_hash(raw_bytes),
        filename=filename,
    )
    
    try:
        import cv2
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=f'.{ext or "mp4"}', delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        
        try:
            cap = cv2.VideoCapture(tmp_path)
            metadata.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            metadata.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            metadata.fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if metadata.fps and frame_count:
                metadata.duration_seconds = frame_count / metadata.fps
            cap.release()
            
            # Estimate tokens (sample frames for vision)
            if metadata.duration_seconds and metadata.width and metadata.height:
                # Sample 1 frame per second
                frames = int(metadata.duration_seconds)
                tokens_per_frame = estimate_image_tokens(metadata.width, metadata.height)
                metadata.estimated_tokens = frames * tokens_per_frame
        finally:
            os.unlink(tmp_path)
            
    except ImportError:
        pass  # opencv not installed
    except Exception:
        pass
    
    return metadata


def analyze_media(data: Union[bytes, str, Path], media_type: str = None) -> Optional[MediaMetadata]:
    """
    Auto-detect and analyze media.
    
    Args:
        data: Media bytes, file path, base64 string, or URL
        media_type: Optional hint ("image", "audio", "video")
        
    Returns:
        MediaMetadata with extracted information
    """
    # Try to detect type from extension or hint
    if media_type:
        if media_type == "image":
            return analyze_image(data)
        elif media_type == "audio":
            return analyze_audio(data)
        elif media_type == "video":
            return analyze_video(data)
    
    # Auto-detect from data
    raw_bytes, filename, ext = _get_bytes(data)
    
    if ext:
        image_exts = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff"}
        audio_exts = {"mp3", "wav", "ogg", "flac", "m4a", "aac"}
        video_exts = {"mp4", "avi", "mov", "mkv", "webm", "wmv"}
        
        if ext in image_exts:
            return analyze_image(data)
        elif ext in audio_exts:
            return analyze_audio(data)
        elif ext in video_exts:
            return analyze_video(data)
    
    # Try image first (most common for AI)
    result = analyze_image(data)
    if result and result.width:
        return result
    
    # Default to unknown
    return MediaMetadata(
        media_type="unknown",
        format=ext or "unknown",
        size_bytes=len(raw_bytes) if raw_bytes else 0,
        content_hash=_compute_hash(raw_bytes) if raw_bytes else None,
    )
