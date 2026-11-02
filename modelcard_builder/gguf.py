"""Minimal, dependency-free GGUF metadata reader.

Reads only the header + key/value metadata block (never tensor data), so it is
fast and safe even on multi-GB files.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

GGUF_MAGIC = b"GGUF"
SUPPORTED_VERSIONS = (1, 2, 3)

SCALAR_FMTS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 10: "<Q", 11: "<q", 12: "<d",
}
SCALAR_SIZES = {k: struct.calcsize(v) for k, v in SCALAR_FMTS.items()}

FILE_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_1 (some F16)",
    7: "Q8_0", 8: "Q8_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
    13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
    18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
    23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
    33: "MXFP4 (MoE)",
}

QUANT_FILE_TYPES = {
    2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33,
}

BPW = {
    "F32": 32.0, "F16": 16.0, "BF16": 16.0,
    "Q8_0": 8.5, "Q8_1": 8.5,
    "Q6_K": 6.56, "Q5_K_M": 5.67, "Q5_K_S": 5.54, "Q5_0": 5.54, "Q5_1": 6.0,
    "Q4_K_M": 4.85, "Q4_K_S": 4.58, "Q4_0": 4.55, "Q4_1": 5.0,
    "Q3_K_M": 3.91, "Q3_K_S": 3.5, "Q3_K_L": 4.27,
    "IQ3_M": 3.66, "IQ3_S": 3.44, "IQ3_XS": 3.3, "IQ3_XXS": 3.06,
    "Q2_K": 3.35, "Q2_K_S": 2.78, "IQ2_M": 2.7, "IQ2_S": 2.5,
    "IQ2_XS": 2.31, "IQ2_XXS": 2.06, "IQ1_S": 1.56, "IQ1_M": 1.75,
    "IQ4_NL": 4.5, "IQ4_XS": 4.25, "MXFP4 (MoE)": 4.25,
}


class GgufError(Exception):
    pass


@dataclass
class ModelInfo:
    name: str
    file_path: Path
    file_size: int
    quantization: str = "unknown"
    architecture: str = "unknown"
    context_length: int | None = None
    embedding_length: int | None = None
    block_count: int | None = None
    expert_count: int | None = None
    parameter_count: int | None = None
    size_label: str | None = None
    chat_template: str | None = None
    version: str | None = None
    finetune: str | None = None
    source: str = "gguf"
    extras: dict = field(default_factory=dict)


class _Reader:
    def __init__(self, fh):
        self.fh = fh
        self.pos = 0

    def bytes(self, n: int) -> bytes:
        data = self.fh.read(n)
        if len(data) != n:
            raise GgufError("unexpected end of file while reading GGUF metadata")
        self.pos += n
        return data

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.bytes(size))[0]


def _read_string(r: _Reader) -> str:
    length = r.unpack("<Q")
    if length > 64 * 1024 * 1024:
        raise GgufError(f"unreasonably long string ({length} bytes)")
    return r.bytes(length).decode("utf-8", errors="replace")


def _read_value(r: _Reader, vtype: int, _depth: int = 0):
    if vtype in (7,):
        return bool(r.unpack("<B"))
    if vtype in SCALAR_FMTS:
        return r.unpack(SCALAR_FMTS[vtype])
    if vtype == 8:
        return _read_string(r)
    if vtype == 9:
        if _depth > 2:
            raise GgufError("nested arrays too deep")
        elem_type = r.unpack("<I")
        count = r.unpack("<Q")
        if elem_type in SCALAR_SIZES:
            r.bytes(count * SCALAR_SIZES[elem_type])
            return None
        if elem_type == 8:
            kept = []
            for i in range(count):
                s = _read_string(r)
                if i < 64:
                    kept.append(s)
            return kept if count <= 64 else kept + [f"({count - 64} more omitted)"]
        raise GgufError("array elements must be scalars or strings")
    raise GgufError(f"unknown GGUF value type {vtype}")


def read_gguf(path: str | os.PathLike) -> ModelInfo:
    p = Path(path)
    with open(p, "rb") as fh:
        r = _Reader(fh)
        magic = r.bytes(4)
        if magic != GGUF_MAGIC:
            raise GgufError(f"{p} is not a GGUF file (bad magic)")
        version = r.unpack("<I")
        if version not in SUPPORTED_VERSIONS:
            raise GgufError(f"unsupported GGUF version {version}")
        tensor_count = r.unpack("<Q")
        kv_count = r.unpack("<Q")

        kv: dict = {}
        for _ in range(kv_count):
            key = _read_string(r)
            vtype = r.unpack("<I")
            kv[key] = _read_value(r, vtype)

    file_size = p.stat().st_size
    arch = str(kv.get("general.architecture", "unknown"))
    file_type = kv.get("general.file_type")
    quant = FILE_TYPE_NAMES.get(int(file_type), "unknown") if file_type is not None else "unknown"
    ctx_key = f"{arch}.context_length"

    name = kv.get("general.name") or infer_name_from_parts(p)
    if isinstance(name, str) and "," in name:
        name = name.split(",")[0]

    ctx = kv.get(ctx_key)
    if not isinstance(ctx, int):
        ctx = None

    info = ModelInfo(
        name=name or p.stem,
        file_path=p,
        file_size=file_size,
        quantization=quant,
        architecture=str(arch),
        context_length=ctx,
        embedding_length=kv.get(f"{arch}.embedding_length") or None,
        block_count=kv.get(f"{arch}.block_count") or None,
        expert_count=kv.get(f"{arch}.expert_count") or None,
        size_label=kv.get("general.size_label") or None,
        chat_template=kv.get("tokenizer.chat_template") if isinstance(kv.get("tokenizer.chat_template"), str) else None,
        version=kv.get("general.version") or None,
        finetune=kv.get("general.finetune") or None,
        extras={"gguf_version": version, "tensor_count": tensor_count},
    )

    if info.quantization == "unknown":
        guess = _guess_quant_from_name(p.name)
        if guess:
            info.extras["quantization_source"] = "filename"
            info.quantization = guess
        else:
            info.extras["quantization_source"] = "none"
    else:
        info.extras["quantization_source"] = "file_type"
    info.extras["kv_metadata"] = {
        k: v for k, v in kv.items()
        if not k.startswith("tokenizer.") and isinstance(v, (int, float, bool, str))
    }
    return info


def _guess_quant_from_name(fname: str) -> str | None:
    for marker in FILE_TYPE_NAMES.values():
        if marker.startswith(("F", "B")):
            continue
        token = marker.split(" ")[0]
        for variant in (token, token + "_K_M", token + "_K_S", token + "_K_L"):
            if variant.replace(" ", "") and variant in fname:
                return variant
        if f".{token}." in fname or fname.endswith(token):
            return token
    return None


def infer_name_from_parts(p: Path) -> str | None:
    stem = p.stem
    for sep in ("--", "-"):
        idx = stem.find(sep)
        if idx > 0:
            return stem[:idx].replace("-", " ") if stem[:idx].isalnum() or " " in stem else stem[:idx]
    return stem


def estimate_parameters(info: ModelInfo) -> int | None:
    if info.parameter_count:
        return info.parameter_count
    label = (info.size_label or "").strip()
    if label and label[-1] in "BM":
        try:
            return int(float(label[:-1]) * (1e9 if label[-1] == "B" else 1e6))
        except ValueError:
            pass
    bpw = BPW.get(info.quantization)
    if bpw and info.file_size > 0:
        return int(info.file_size / (bpw / 8) * 0.90)
    return None


def human_size(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:,.1f} GB" if unit in ("GB",) else f"{f:,.1f} {unit}" if unit != "B" else f"{int(f)} B"
        f /= 1024
    return f"{f:,.1f} TB"
