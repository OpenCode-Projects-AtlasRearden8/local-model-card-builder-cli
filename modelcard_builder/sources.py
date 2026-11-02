"""Source adapters: Ollama and LM Studio model directories."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from .gguf import ModelInfo, GgufError, read_gguf, infer_name_from_parts


class SourceError(Exception):
    pass


def ollama_list() -> list[str]:
    try:
        proc = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=30.0
        )
    except FileNotFoundError as exc:
        raise SourceError("Ollama is not installed (or not on PATH).") from exc
    except subprocess.TimeoutExpired as exc:
        raise SourceError("ollama list timed out") from exc
    if proc.returncode != 0:
        raise SourceError(proc.stderr.strip() or "ollama list failed")
    models = []
    for line in proc.stdout.strip().splitlines()[1:]:
        if line.strip():
            models.append(line.split()[0])
    return models


def ollama_model_info(model: str) -> ModelInfo:
    try:
        proc = subprocess.run(
            ["ollama", "show", model, "--json"],
            capture_output=True, text=True, timeout=30.0,
        )
    except FileNotFoundError as exc:
        raise SourceError("Ollama is not installed (or not on PATH).") from exc
    except subprocess.TimeoutExpired as exc:
        raise SourceError("ollama show timed out") from exc
    if proc.returncode != 0:
        raise SourceError(proc.stderr.strip() or f"ollama show {model} failed")
    raw = proc.stdout.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SourceError("ollama did not return JSON; try upgrading Ollama") from exc

    details = data.get("details", {}) or {}
    model_info_keys = data.get("model_info") or {}
    arch = str(
        model_info_keys.get("general.architecture")
        or ",".join(details.get("families") or ["unknown"]) or "unknown"
    )

    size_bytes = data.get("model_size") or data.get("size") or details.get("size") or 0
    if not size_bytes:
        size_bytes = _ollama_size_from_list(model)

    quant = str(
        details.get("quantization")
        or model_info_keys.get("general.file_type_label")
        or _quant_from_file_type(model_info_keys.get("general.file_type"))
        or "default (from Ollama: usually Q4_K_M-class)"
    )

    path_hint = data.get("model_path") or data.get("path")
    file_path = Path(path_hint) if path_hint else Path(os.environ.get("OLLAMA_MODELS", "~/.ollama/models")).expanduser() / "blobs"

    ctx = 0
    for key, val in model_info_keys.items():
        if key.endswith(".context_length") and isinstance(val, int):
            ctx = max(ctx, val)

    return ModelInfo(
        name=model,
        file_path=file_path,
        file_size=int(size_bytes or 0),
        quantization=quant,
        architecture=arch,
        context_length=ctx or None,
        block_count=model_info_keys.get(f"{arch}.block_count"),
        size_label=details.get("parameter_size") or None,
        chat_template=str(data.get("template") or "") or None,
        source=f"ollama ({model})",
        extras={"capabilities": details.get("capabilities", [])},
    )


def _quant_from_file_type(file_type) -> str | None:
    from .gguf import FILE_TYPE_NAMES

    if isinstance(file_type, int):
        return FILE_TYPE_NAMES.get(file_type)
    if isinstance(file_type, str) and file_type:
        return file_type
    return None


def _ollama_size_from_list(model: str) -> int:
    try:
        proc = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=30.0
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    for line in proc.stdout.strip().splitlines()[1:]:
        parts = line.split()
        if parts and parts[0] == model and len(parts) >= 3:
            return _parse_human_size(" ".join(parts[2:4]))
    return 0


def _parse_human_size(text: str) -> int:
    m = re.match(r"([\d.]+)\s*(B|KB|MB|GB|TB)", text, re.IGNORECASE)
    if not m:
        return 0
    value, unit = float(m.group(1)), m.group(2).upper()
    mult = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}[unit]
    return int(value * mult)


def lmstudio_dirs() -> list[Path]:
    explicit = os.environ.get("LMSTUDIO_MODELS_DIR")
    dirs = []
    if explicit:
        dirs.append(Path(explicit))
    else:
        home = Path.home()
        dirs += [
            home / ".lmstudio" / "models",
            home / ".cache" / "lm-studio" / "models",
            home / "Library" / "Application Support" / "lm-studio" / "models",
        ]
    return [d for d in dirs if d.is_dir()]


def lmstudio_models(models_dir: Path | str | None = None) -> list[ModelInfo]:
    if models_dir is not None:
        models_dir = Path(models_dir)
    if models_dir is None:
        found = lmstudio_dirs()
        if not found:
            raise SourceError(
                "LM Studio models directory not found. Pass --dir or set LMSTUDIO_MODELS_DIR."
            )
        models_dir = found[0]

    infos: list[ModelInfo] = []
    for gguf in sorted(models_dir.rglob("*.gguf")):
        try:
            info = read_gguf(gguf)
        except (GgufError, OSError) as exc:
            info = ModelInfo(
                name=infer_name_from_parts(gguf) or gguf.stem,
                file_path=gguf,
                file_size=gguf.stat().st_size,
                source=f"lmstudio ({exc})",
            )
        info.source = f"lmstudio ({info.file_path.parent.relative_to(models_dir)})"
        infos.append(info)
    return infos
