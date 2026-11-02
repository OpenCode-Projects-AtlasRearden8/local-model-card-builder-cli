import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gguf_sample import llama_kvs, write_gguf

from modelcard_builder import cli


def run_cli(argv, cwd=None):
    """Run the CLI, capturing output; cwd is swapped in temporarily."""
    out, err = io.StringIO(), io.StringIO()
    original = Path.cwd()
    if cwd:
        import os

        os.chdir(cwd)
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(argv)
    finally:
        if cwd:
            import os

            os.chdir(original)
    return code, out.getvalue(), err.getvalue()


class TestCliGguf(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.gguf = self.dir / "testllama-v3.1-Q4_K_M.gguf"
        write_gguf(self.gguf, llama_kvs())

    def test_gguf_writes_card(self):
        code, out, err = run_cli(["gguf", str(self.gguf), "-o", "card.md"], cwd=self.dir)
        self.assertEqual(code, 0)
        path = self.dir / "card.md"
        self.assertTrue(path.exists())
        self.assertIn("Model card: TestLlama", path.read_text())
        self.assertIn("wrote card.md", out)

    def test_gguf_default_filename(self):
        code, out, _ = run_cli(["gguf", str(self.gguf)], cwd=self.dir)
        self.assertEqual(code, 0)
        default = cli.card_filename(
            __import__("modelcard_builder.gguf", fromlist=["read_gguf"]).read_gguf(self.gguf)
        )
        self.assertTrue((self.dir / default).exists())

    def test_gguf_bad_file(self):
        bad = self.dir / "bad.gguf"
        bad.write_bytes(b"oops")
        code, _, err = run_cli(["gguf", str(bad)], cwd=self.dir)
        self.assertEqual(code, 1)
        self.assertIn("error:", err)


class TestCliOther(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_ollama_without_cli_fails_cleanly(self):
        code, _, err = run_cli(["ollama", "m"], cwd=self.dir)
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

    def test_lmstudio_list(self):
        sub = self.dir / "models"
        sub.mkdir()
        write_gguf(sub / "m-Q4_K_M.gguf", llama_kvs())
        code, out, _ = run_cli(["lmstudio", "--dir", str(sub), "--list"], cwd=self.dir)
        self.assertEqual(code, 0)
        self.assertIn("m-Q4_K_M.gguf", out)

    def test_lmstudio_writes_card_from_selected_file(self):
        sub = self.dir / "models"
        sub.mkdir()
        write_gguf(sub / "m-Q4_K_M.gguf", llama_kvs())
        code, out, _ = run_cli(["lmstudio", "--dir", str(sub), "-o", "lms.md"], cwd=self.dir)
        self.assertEqual(code, 0)
        self.assertIn("Model card: TestLlama", (self.dir / "lms.md").read_text())

    def test_scan_runs_without_gguf_hits(self):
        code, out, _ = run_cli(["scan"], cwd=self.dir)
        self.assertEqual(code, 0)

    def test_config_flags(self):
        code, out, _ = run_cli(["config"], cwd=self.dir)
        self.assertEqual(code, 0)
        self.assertIn("polish endpoint", out)
        code, _, _ = run_cli(["config", "--init"], cwd=self.dir)
        self.assertEqual(code, 0)
        self.assertTrue((self.dir / "modelcard.toml").exists())
        code, _, err = run_cli(["config", "--init"], cwd=self.dir)
        self.assertEqual(code, 1)

    def test_no_command_shows_help(self):
        code, out, _ = run_cli([], cwd=self.dir)
        self.assertEqual(code, 0)
        self.assertIn("usage:", out)


if __name__ == "__main__":
    unittest.main()
