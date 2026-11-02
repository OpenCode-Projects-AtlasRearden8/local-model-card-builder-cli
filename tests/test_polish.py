import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from modelcard_builder.config import Config
from modelcard_builder.gguf import ModelInfo
from modelcard_builder.polish import PolishError, polish_text, build_prompt, fact_sheet
from pathlib import Path


class Handler(BaseHTTPRequestHandler):
    behavior = {"status": 200, "body": {}}

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length))
        for check, msg in Handler.behavior.get("require", []):
            if not check(data):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(msg.encode())
                return
        self.send_response(Handler.behavior["status"])
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(Handler.behavior["body"]).encode())

    def log_message(self, *args):
        pass


class TestPolish(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        Handler.behavior = {
            "status": 200,
            "body": {"choices": [{"message": {"content": " In a nutshell: friendly words. "}}]},
        }

    def _cfg(self, **kw) -> Config:
        return Config(endpoint=f"http://127.0.0.1:{self.port}/v1", timeout=10.0, **kw)
    def test_happy_path_trims(self):
        out = polish_text("Model name: x", self._cfg())
        self.assertEqual(out, "In a nutshell: friendly words.")

    def test_endpoint_and_model_forwarded(self):
        Handler.behavior["require"] = [
            (lambda d: d["model"] == "testmodel", "model missing"),
            (lambda d: d["messages"][0]["role"] == "system", "system missing"),
        ]
        polish_text("Model name: x", self._cfg(polish_model="testmodel"))

    def test_http_error_raises_polish_error(self):
        Handler.behavior["status"] = 500
        with self.assertRaises(PolishError):
            polish_text("Model name: x", self._cfg())

    def test_bad_shape_raises(self):
        Handler.behavior["body"] = {"nope": True}
        with self.assertRaises(PolishError):
            polish_text("Model name: x", self._cfg())

    def test_empty_content_raises(self):
        Handler.behavior["body"] = {"choices": [{"message": {"content": ""}}]}
        with self.assertRaises(PolishError):
            polish_text("Model name: x", self._cfg())

    def test_error_field_raises(self):
        Handler.behavior["body"] = {"error": "model not loaded"}
        with self.assertRaises(PolishError):
            polish_text("Model name: x", self._cfg())

    def test_bad_endpoint_rejected_early(self):
        with self.assertRaises(PolishError):
            polish_text("Model name: x", Config(endpoint="localhost:11434"))

    def test_connection_refused_raises_polish_error(self):
        with self.assertRaises(PolishError):
            polish_text("Model name: x", Config(endpoint="http://127.0.0.1:1/v1", timeout=2.0))

    def test_build_prompt_messages(self):
        messages = build_prompt("Model name: x")
        self.assertEqual(len(messages), 2)
        self.assertIn("friend", messages[0]["content"].lower())

    def test_fact_sheet_fields(self):
        info = ModelInfo(
            name="TestLlama", file_path=Path("x.gguf"), file_size=4_000_000_000,
            quantization="Q4_K_M", architecture="llama", context_length=8192,
            chat_template="yes",
        )
        text = fact_sheet(info)
        self.assertIn("TestLlama", text)
        self.assertIn("Q4_K_M", text)
        self.assertIn("8,192", text)


if __name__ == "__main__":
    unittest.main()
