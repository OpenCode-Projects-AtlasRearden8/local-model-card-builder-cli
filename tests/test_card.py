import tempfile
import unittest
from pathlib import Path

from gguf_sample import llama_kvs, write_gguf

from modelcard_builder.card import render_card, card_filename
from modelcard_builder.gguf import read_gguf


class TestCard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "testllama-v3.1-Q4_K_M.gguf"
        write_gguf(self.path, llama_kvs())
        self.info = read_gguf(self.path)

    def test_card_contains_key_sections(self):
        text = render_card(self.info)
        for heading in (
            "Model card: TestLlama",
            "Quantization",
            "Practical use-cases",
            "Prompt style notes",
            "My notes",
        ):
            self.assertIn(heading, text)
        self.assertIn("Q4_K_M", text)
        self.assertIn("8,192 tokens", text)
        self.assertIn("32 pages of plain text", text)

    def test_polish_section_optional(self):
        self.assertNotIn("LLM polish", render_card(self.info))
        self.assertIn("LLM polish", render_card(self.info, polished="Warm words."))
        self.assertIn("Warm words.", render_card(self.info, polished="Warm words.\n"))

    def test_quant_guidance_ends_plain(self):
        self.assertIn("notice board", render_card(self.info))

    def test_card_filename_sane(self):
        name = card_filename(self.info)
        self.assertTrue(name.endswith(".md"))
        self.assertIn("testllama", name.lower())

    def test_no_chat_template_note(self):
        self.path = self.dir / "plain-Q6_K.gguf"
        write_gguf(self.path, llama_kvs(chat_template=None))
        info = read_gguf(self.path)
        text = render_card(info)
        self.assertIn("No built-in chat template", text)


if __name__ == "__main__":
    unittest.main()
