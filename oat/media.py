"""Multimodal extraction helpers used by integrations and legacy ingest."""

from __future__ import annotations

from typing import Any, Dict, List
import base64


def _guess_data_url_mime(value: str, default: str) -> str:
    if not isinstance(value, str):
        return default
    if value.startswith("data:") and ";" in value:
        return value.split(";", 1)[0].replace("data:", "") or default
    return default


def _estimate_data_url_size(value: str) -> int:
    if not isinstance(value, str) or "," not in value or not value.startswith("data:"):
        return 0
    payload = value.split(",", 1)[1]
    try:
        return len(base64.b64decode(payload, validate=False))
    except Exception:
        return 0


def extract_openai_multimodal_artifacts(messages: Any) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    if not isinstance(messages, list):
        return artifacts

    for mi, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            artifacts.append(
                {
                    "role": "input.text",
                    "content_type": "text/plain",
                    "content": content,
                    "inline_text": content,
                    "preview": {"message_index": mi},
                    "metadata": {"message_index": mi},
                }
            )
            continue

        if not isinstance(content, list):
            continue
        for ci, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            ptype = str(part.get("type") or "")

            if ptype == "text" and part.get("text"):
                text = str(part.get("text"))
                artifacts.append(
                    {
                        "role": "input.text",
                        "content_type": "text/plain",
                        "content": text,
                        "inline_text": text,
                        "preview": {"message_index": mi, "part_index": ci},
                        "metadata": {"message_index": mi, "part_index": ci},
                    }
                )
                continue

            if ptype == "image_url":
                img = part.get("image_url") or {}
                url = img.get("url")
                if not url:
                    continue
                ctype = _guess_data_url_mime(str(url), "image/*")
                artifacts.append(
                    {
                        "role": "input.image",
                        "content_type": ctype,
                        "content": url,
                        "preview": {"message_index": mi, "part_index": ci},
                        "metadata": {
                            "message_index": mi,
                            "part_index": ci,
                            "image_detail": img.get("detail"),
                            "size_bytes_estimate": _estimate_data_url_size(str(url)),
                        },
                    }
                )
                continue

            if ptype in {"input_audio", "audio", "audio_url"}:
                audio = part.get("input_audio") or part.get("audio") or part.get("audio_url") or {}
                if isinstance(audio, str):
                    audio = {"url": audio}
                url = audio.get("url")
                data = audio.get("data")
                fmt = audio.get("format") or "unknown"
                content = data or url
                if not content:
                    continue
                ctype = f"audio/{fmt}" if fmt != "unknown" else _guess_data_url_mime(str(content), "audio/*")
                artifacts.append(
                    {
                        "role": "input.audio",
                        "content_type": ctype,
                        "content": content,
                        "preview": {"message_index": mi, "part_index": ci},
                        "metadata": {
                            "message_index": mi,
                            "part_index": ci,
                            "format": fmt,
                            "size_bytes_estimate": _estimate_data_url_size(str(content)),
                        },
                    }
                )
                continue

            if ptype in {"input_audio_transcript", "audio_transcript"}:
                text = part.get("text") or part.get("transcript")
                if not text:
                    continue
                artifacts.append(
                    {
                        "role": "derived.audio_transcript",
                        "content_type": "text/plain",
                        "content": str(text),
                        "inline_text": str(text),
                        "preview": {"message_index": mi, "part_index": ci},
                        "metadata": {"message_index": mi, "part_index": ci},
                    }
                )
    return artifacts


def extract_anthropic_multimodal_artifacts(messages: Any) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    if not isinstance(messages, list):
        return artifacts
    for mi, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            artifacts.append(
                {
                    "role": "input.text",
                    "content_type": "text/plain",
                    "content": content,
                    "inline_text": content,
                    "preview": {"message_index": mi},
                    "metadata": {"message_index": mi},
                }
            )
            continue
        if not isinstance(content, list):
            continue
        for ci, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            ptype = str(part.get("type") or "")
            if ptype == "text" and part.get("text"):
                text = str(part["text"])
                artifacts.append(
                    {
                        "role": "input.text",
                        "content_type": "text/plain",
                        "content": text,
                        "inline_text": text,
                        "preview": {"message_index": mi, "part_index": ci},
                        "metadata": {"message_index": mi, "part_index": ci},
                    }
                )
                continue
            if ptype == "image":
                source = part.get("source") or {}
                media_type = source.get("media_type") or "image/*"
                payload = source.get("data") or source.get("url")
                if not payload:
                    continue
                artifacts.append(
                    {
                        "role": "input.image",
                        "content_type": media_type,
                        "content": payload,
                        "preview": {"message_index": mi, "part_index": ci},
                        "metadata": {"message_index": mi, "part_index": ci, "source_type": source.get("type")},
                    }
                )
            if ptype in {"audio", "input_audio"}:
                source = part.get("source") or {}
                media_type = source.get("media_type") or "audio/*"
                payload = source.get("data") or source.get("url")
                if not payload:
                    continue
                artifacts.append(
                    {
                        "role": "input.audio",
                        "content_type": media_type,
                        "content": payload,
                        "preview": {"message_index": mi, "part_index": ci},
                        "metadata": {"message_index": mi, "part_index": ci, "source_type": source.get("type")},
                    }
                )
    return artifacts


def extract_gemini_multimodal_artifacts(contents: Any) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    if not isinstance(contents, list):
        contents = [contents] if contents is not None else []
    for pi, content in enumerate(contents):
        parts = None
        if isinstance(content, dict):
            parts = content.get("parts")
        elif hasattr(content, "parts"):
            parts = getattr(content, "parts")
        if parts is None:
            parts = [content]
        if not isinstance(parts, list):
            parts = [parts]

        for ci, part in enumerate(parts):
            data = part
            if hasattr(part, "to_dict"):
                try:
                    data = part.to_dict()
                except Exception:
                    data = part
            if isinstance(data, str):
                artifacts.append(
                    {
                        "role": "input.text",
                        "content_type": "text/plain",
                        "content": data,
                        "inline_text": data,
                        "preview": {"part_index": pi, "content_index": ci},
                        "metadata": {"part_index": pi, "content_index": ci},
                    }
                )
                continue
            if not isinstance(data, dict):
                continue

            if data.get("text"):
                text = str(data["text"])
                artifacts.append(
                    {
                        "role": "input.text",
                        "content_type": "text/plain",
                        "content": text,
                        "inline_text": text,
                        "preview": {"part_index": pi, "content_index": ci},
                        "metadata": {"part_index": pi, "content_index": ci},
                    }
                )
            if data.get("inline_data"):
                inline = data.get("inline_data") or {}
                ctype = inline.get("mime_type") or "application/octet-stream"
                payload = inline.get("data")
                if payload:
                    role = "input.image" if str(ctype).startswith("image/") else "input.audio" if str(ctype).startswith("audio/") else "input.message.binary"
                    artifacts.append(
                        {
                            "role": role,
                            "content_type": ctype,
                            "content": payload,
                            "preview": {"part_index": pi, "content_index": ci},
                            "metadata": {"part_index": pi, "content_index": ci},
                        }
                    )
            if data.get("file_data"):
                f = data.get("file_data") or {}
                ctype = f.get("mime_type") or "application/octet-stream"
                uri = f.get("file_uri")
                if uri:
                    role = "input.image" if str(ctype).startswith("image/") else "input.audio" if str(ctype).startswith("audio/") else "input.message.binary"
                    artifacts.append(
                        {
                            "role": role,
                            "content_type": ctype,
                            "content": uri,
                            "preview": {"part_index": pi, "content_index": ci},
                            "metadata": {"part_index": pi, "content_index": ci},
                        }
                    )
    return artifacts
