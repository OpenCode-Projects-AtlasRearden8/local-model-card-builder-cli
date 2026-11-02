import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from gguf_sample import write_gguf, llama_kvs

from modelcard_builder.gguf import ModelInfo
from modelcard_builder.sources import (
    SourceError,
    _parse_human_size,
    _quant_from_file_type,
    lmstudio_models,
    ollama_list,
    ollama_model_info,
)


def _fake_run(stdout, returncode=0):
    proc = subprocess.CompletedProcess(["ollama"], returncode)
    proc.stdout = stdout
    proc.stderr = "" if returncode == 0 else "boom"
    return proc


class TestOllamaList(unittest.TestCase):
    @mock.patch("modelcard_builder.sources.subprocess.run")
    def test_parses_list(self, run):
        run.return_value = _fake_run(
            "NAME        ID          SIZE      MODIFIED\n"
            "llama3.2    abc         2.0 GB    3 days ago\n"
            "qwen2.5:7b  def         4.4 GB    1 hour ago\n"
        )
        self.assertEqual(ollama_list(), ["llama3.2", "qwen2.5:7b"])

    @mock.patch("modelcard_builder.sources.subprocess.run")
    def test_missing_binary(self, run):
        run.side_effect = FileNotFoundError()
        with self.assertRaises(SourceError):
            ollama_list()

    @mock.patch("modelcard_builder.sources.subprocess.run")
    def test_bad_returncode(self, run):
        run.return_value = _fake_run("", returncode=1)
        with self.assertRaises(SourceError):
            ollama_list()


class TestOllamaModelInfo(unittest.TestCase):
    payload = {
        "model_path": "/tmp/blobs/sha256-abc",
        "model_size": 4_000_000_000,
        "template": "{{ .prompt }}",
        "details": {
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "8B",
            "quantization": "Q4_K_M",
            "capabilities": ["chat"],
        },
        "model_info": {
            "general.architecture": "llama",
            "llama.context_length": 131072,
            "general.file_type": 15,
            "llama.block_count": 32,
        },
    }

    @mock.patch("modelcard_builder.sources.subprocess.run")
    def test_parses_show_json(self, run):
        run.return_value = _fake_run(json.dumps(self.payload))
        info = ollama_model_info("llama3.2")
        self.assertEqual(info.name, "llama3.2")
        self.assertEqual(info.quantization, "Q4_K_M")
        self.assertEqual(info.context_length, 131072)
        self.assertEqual(info.file_size, 4_000_000_000)
        self.assertEqual(info.source, "ollama (llama3.2)")
        self.assertTrue(info.chat_template)

    @mock.patch("modelcard_builder.sources.subprocess.run")
    def test_size_falls_back_to_list(self, run):
        payload = dict(self.payload)
        payload.pop("model_size")
        listing = _fake_run(
            "NAME          ID   SIZE     MODIFIED\nllama3.2      abc  3.5 GB   now\n"
        )

        def side_effect(cmd, **kw):
            return listing if cmd[-1] == "list" else _fake_run(json.dumps(payload))

        run.side_effect = side_effect
        info = ollama_model_info("llama3.2")
        self.assertEqual(info.file_size, 3_500_000_000)

    @mock.patch("modelcard_builder.sources.subprocess.run")
    def test_broken_json(self, run):
        run.return_value = _fake_run("not json at all")
        with self.assertRaises(SourceError):
            ollama_model_info("m")

    def test_quant_from_file_type(self):
        self.assertEqual(_quant_from_file_type(15), "Q4_K_M")
        self.assertEqual(_quant_from_file_type("Q8_0"), "Q8_0")
        self.assertIsNone(_quant_from_file_type(None))

    def test_parse_human_size(self):
        self.assertEqual(_parse_human_size("2.0 GB"), 2_000_000_000)
        self.assertEqual(_parse_human_size("512 MB"), 512_000_000)
        self.assertEqual(_parse_human_size("junk"), 0)


class TestLmStudio(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_finds_and_reads_ggufs(self):
        sub = self.dir / "TheBloke" / "Llama-Q4_K_M-GGUF"
        sub.mkdir(parents=True)
        write_gguf(sub / "llama-v2-Q4_K_M.gguf", llama_kvs())
        (self.dir / "junk-not-gguf.bin").write_bytes(b"nothing")
        infos = lmstudio_models(self.dir)
        paths = [i.file_path.name for i in infos]
        self.assertEqual(paths, ["llama-v2-Q4_K_M.gguf"])
        self.assertEqual(infos[0].quantization, "Q4_K_M")
        self.assertTrue(infos[0].source.startswith("lmstudio"))

    def test_unreadable_gguf_gets_placeholder(self):
        (self.dir / "broken.gguf").write_bytes(b"bad")
        infos = lmstudio_models(self.dir)
        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0].quantization, "unknown")


if __name__ == "__main__":
    unittest.main()
