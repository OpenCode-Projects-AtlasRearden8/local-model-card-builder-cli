"""Write a minimal but valid GGUF file for tests.

Only what the header parser needs: magic, version, tensor/kv counts, and
key/value metadata. No tensor data is written (the reader never reads it).
"""
import struct


def _kv_str(key: str, value: str) -> bytes:
    k = struct.pack("<Q", len(key.encode())) + key.encode()
    v = value.encode()
    return k + struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v


def _kv_num(key: str, value, vtype: int, fmt: str) -> bytes:
    k = struct.pack("<Q", len(key.encode())) + key.encode()
    return k + struct.pack("<I", vtype) + struct.pack(fmt, value)


def write_gguf(path, kvs: dict, version: int = 3) -> None:
    out = b"GGUF" + struct.pack("<I", version)
    out += struct.pack("<Q", 0)          # tensor_count
    out += struct.pack("<Q", len(kvs))   # kv_count
    for key, (value, vtype, fmt) in kvs.items():
        if fmt == "str":
            out += _kv_str(key, value)
        elif fmt == "bool":
            out += (
                struct.pack("<Q", len(key.encode())) + key.encode()
                + struct.pack("<I", 7) + bytes([1 if value else 0])
            )
        else:
            out += _kv_num(key, value, vtype, fmt)
    out += b"\x00" * 16  # pretend tensor data (never read)
    with open(path, "wb") as fh:
        fh.write(out)


def llama_kvs(
    name: str = "TestLlama",
    arch: str = "llama",
    file_type: int = 15,
    context_length: int = 8192,
    chat_template: str | None = "<|user|>{prompt}",
    version: str = "v3.1",
) -> dict:
    kvs = {
        "general.name": (name, 8, "str"),
        "general.architecture": (arch, 8, "str"),
        "general.file_type": (file_type, 4, "<I"),
        "general.quantization_version": (2, 4, "<I"),
        "general.size_label": ("8B", 8, "str"),
        "general.version": (version, 8, "str"),
        f"{arch}.context_length": (context_length, 10, "<Q"),
        f"{arch}.embedding_length": (4096, 10, "<Q"),
        f"{arch}.block_count": (32, 4, "<I"),
    }
    if chat_template is not None:
        kvs["tokenizer.chat_template"] = (chat_template, 8, "str")
    return kvs
