import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from gguf_sample import llama_kvs, write_gguf

from modelcard_builder.gguf import GgufError, read_gguf, human_size, estimate_parameters


class TestGgufReader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_reads_basic_metadata(self):
        path = self.dir / "testllama-v3.1-Q4_K_M.gguf"
        write_gguf(path, llama_kvs(context_length=8192, chat_template="tmpl"))
        info = read_gguf(path)
        self.assertEqual(info.name, "TestLlama")
        self.assertEqual(info.quantization, "Q4_K_M")
        self.assertEqual(info.context_length, 8192)
        self.assertEqual(info.block_count, 32)
        self.assertEqual(info.architecture, "llama")
        self.assertEqual(info.chat_template, "tmpl")
        self.assertEqual(info.size_label, "8B")
        self.assertEqual(info.version, "v3.1")
        self.assertEqual(info.extras["gguf_version"], 3)

    def test_size_and_params_estimate(self):
        path = self.dir / "m.gguf"
        write_gguf(path, {"general.name": ("m", 8, "str")})
        info = read_gguf(path)
        self.assertEqual(human_size(info.file_size), f"{info.file_size:,} B".replace(",", "").replace("B", "B").strip())
        self.assertEqual(estimate_parameters(info), None)

    def test_estimates_params_from_filename_quant_and_size(self):
        path = self.dir / "tiny.gguf"
        write_gguf(path, {"general.name": ("tiny", 8, "str")}, version=2)
        import os

        with open(path, "r+b") as fh:
            fh.seek(12)
            fh.write((2).to_bytes(8, "little"))
        info = read_gguf(path)
        info.quantization = "Q8_0"
        params = estimate_parameters(info)
        self.assertTrue(params and params > 0)

    def test_filename_quant_fallback(self):
        path = self.dir / "somerandom-Q6_K.gguf"
        write_gguf(path, {"general.name": ("somerandom", 8, "str")}, version=2)
        info = read_gguf(path)
        self.assertEqual(info.quantization, "Q6_K")
        self.assertEqual(info.extras["quantization_source"], "filename")

    def test_bad_magic(self):
        path = self.dir / "not.gguf"
        path.write_bytes(b"NOPE" + b"\x00" * 20)
        with self.assertRaises(GgufError):
            read_gguf(path)

    def test_unsupported_version(self):
        path = self.dir / "v99.gguf"
        with open(path, "wb") as fh:
            fh.write(b"GGUF")
            fh.write((99).to_bytes(4, "little"))
        with self.assertRaises(GgufError):
            read_gguf(path)

    def test_truncated_raises_cleanly(self):
        path = self.dir / "trunc.gguf"
        with open(path, "wb") as fh:
            fh.write(b"GGUF" + (3).to_bytes(4, "little") + b"\x00" * 10)
        with self.assertRaises(GgufError):
            read_gguf(path)


if __name__ == "__main__":
    unittest.main()
