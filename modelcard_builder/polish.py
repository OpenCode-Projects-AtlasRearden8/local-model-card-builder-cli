"""Optional polish pass through any OpenAI-compatible chat endpoint.

Defaults to a local Ollama server (http://localhost:11434/v1). Only the
stdlib is used; the API key, if a endpoint needs one, is read from an
environment variable and never stored.
"""
from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request

from .config import Config

SYSTEM_PROMPT = (
    "You are a friendly editor. Explain AI like a friend would: plain English, "
    "step by step, never condescending. You will be given a draft model card "
    "about a locally installed AI model. Write a short, warm section titled "
    "'In a nutshell' with 3-5 short sentences: who this model is a good fit "
    "for, what to try first, and one honest caveat. Use no markdown headings "
    "or bullet lists; plain sentences only. Be specific, no hype."
)


class PolishError(Exception):
    pass


def build_prompt(info_summary: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Here are the facts about the model:\n\n"
                f"{info_summary}\n\n"
                "Write the 'In a nutshell' paragraph now."
            ),
        },
    ]


def polish_text(info_summary: str, cfg: Config | None = None) -> str:
    cfg = cfg or Config()
    if not cfg.endpoint.lower().startswith(("http://", "https://")):
        raise PolishError(f"endpoint must be an http(s) URL, got: {cfg.endpoint}")

    body = json.dumps(
        {
            "model": cfg.polish_model,
            "messages": build_prompt(info_summary),
            "temperature": 0.6,
            "stream": False,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        cfg.chat_url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "modelcard-builder/0.1"},
        method="POST",
    )
    api_key = os.environ.get(cfg.api_key_env or "MC_API_KEY", "")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")

    context = None
    if cfg.endpoint.lower().startswith("https://"):
        context = ssl.create_default_context()

    try:
        with urllib.request.urlopen(request, timeout=cfg.timeout, context=context) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(400).decode("utf-8", errors="replace")
        raise PolishError(f"polish endpoint returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise PolishError(f"could not reach polish endpoint {cfg.endpoint}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PolishError("polish endpoint returned invalid JSON") from exc

    try:
        if payload.get("error"):
            raise PolishError(f"polish endpoint error: {payload['error']}")
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise PolishError(f"unexpected response shape from endpoint: {exc}") from exc
    if not text or not str(text).strip():
        raise PolishError("endpoint returned empty content")
    return str(text).strip()


def fact_sheet(info) -> str:
    from .gguf import human_size, estimate_parameters, BPW

    params = estimate_parameters(info)
    parts = [
        f"Model name: {info.name}",
        f"File size: {human_size(info.file_size)}",
        f"Quantization: {info.quantization}"
        + (f" ({BPW.get(info.quantization, 0):.1f} bits per weight)" if BPW.get(info.quantization) else ""),
        f"Architecture: {info.architecture}",
    ]
    if params:
        parts.append(f"Estimated parameters: ~{params / 1e9:.1f}B" if params >= 1e9 else f"Estimated parameters: ~{params / 1e6:.0f}M")
    if info.context_length:
        parts.append(f"Context length: {info.context_length:,} tokens")
    if info.chat_template:
        parts.append("Ships a chat template: yes")
    return "\n".join(str(p) for p in parts)
