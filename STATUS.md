# STATUS — modelcard-builder

**Date:** 2026-11-02 · **License:** MIT · **Copyright (c) 2026 Asdale Dynamics, LLC**

## What was built

`modelcard-builder` — a Python stdlib-only CLI that turns local LLM metadata
into a readable, beginner-friendly **model card in Markdown**.

Files:

- `modelcard_builder/`
  - `gguf.py` — dependency-free GGUF header/metadata reader (magic, version,
    kv block only; never touches tensor data). Understands quantization file
    types (Q4_K_M, Q6_K, IQ-variants, …), estimates parameters from
    `general.size_label` or from bytes-per-weight + file size, reads context
    length, block count, and `tokenizer.chat_template`.
  - `sources.py` — source adapters: `ollama list` / `ollama show --json`
    (with graceful fallbacks when Ollama is missing or old) and LM Studio
    model-directory scanning (auto-detects `~/.lmstudio/models` etc., honors
    `LMSTUDIO_MODELS_DIR` and `--dir`).
  - `card.py` — markdown renderer: at-a-glance facts, plain-English
    quantization guidance per quant family, practical use-cases scaled by
    estimated parameters, honest memory/RAM ranges, prompt style notes
    (chat template present or not, per-family tips), a fill-in "My notes"
    section, and an optional "LLM polish" section.
  - `polish.py` — optional polish pass against any OpenAI-compatible
    `${endpoint}/v1/chat/completions` endpoint (stdlib `urllib`; default is
    local Ollama `http://localhost:11434/v1`, model `llama3.2`).
  - `config.py` — settings precedence: CLI flags → env (`MC_ENDPOINT`,
    `MC_POLISH_MODEL`, `MC_API_KEY`) → `modelcard.toml` (cwd, then
    `~/.config/modelcard-builder/config.toml`) → defaults.
  - `cli.py` — argparse CLI, version 0.1.0.
- `pyproject.toml` — packaging + `modelcard` console script (setuptools).
- `tests/` — 41 unit/CLI/e2e tests (stdlib `unittest`), including a synthetic
  GGUF writer and a stub OpenAI-compatible HTTP server on localhost.
- `README.md`, `LICENSE` (MIT, exact line "Copyright (c) 2026 Asdale
  Dynamics, LLC"), `.gitignore`. No secrets, keys, or personal data anywhere.

## How to run it

```bash
python3 -m modelcard_builder --help            # from this folder, no install
python3 -m modelcard_builder scan              # find local models
python3 -m modelcard_builder gguf model.gguf   # card from a GGUF file
python3 -m modelcard_builder ollama llama3.2   # card from an Ollama model
python3 -m modelcard_builder lmstudio --list   # card from LM Studio files
python3 -m modelcard_builder gguf model.gguf --polish   # + local LLM summary
modelcard ...                                  # after `pip install .`
python3 -m unittest discover -s tests          # test suite
```

## Verification results (all run on this machine, 2026-11-02)

- `python3 -m unittest discover -s tests` → **Ran 41 tests — OK** (0 failures).
- Generated a full card from a synthetic llama-8B-style GGUF (v3 header,
  chat template, context 131,072): all sections render, facts correct
  (Q6_K ≈ 6.6 bits/weight, ~8.0B params from size label, 524-page estimate).
- `scan` on a machine without Ollama/LM Studio: friendly source-by-source
  report, exit 0.
- `ollama` command without Ollama installed: clean error message, exit 1.
- `lmstudio --dir` listing + card generation: verified.
- `pip install .` + `modelcard` console script verified inside a fresh venv
  (system pip is PEP 668 externally managed here, as expected on many Linuxes).
- `config --init`: writes a commented starter `modelcard.toml`.
- `--polish` end-to-end against a local stub OpenAI-compatible server
  (127.0.0.1): request shaped correctly (model, system+user messages),
  polished section inserted into the card.
- `--polish` against a dead endpoint: prints "polish skipped (…)", still
  writes the unpolished card, exit 0.

## Known gaps / honest caveats

- **Ollama needs a recent version** for `ollama show <m> --json`; older
  builds raise a "try upgrading Ollama" message rather than guessing.
- Parameter counts are **estimates** (from `general.size_label` when present,
  else derived from quant bits-per-weight × file size with a 10% overhead
  assumption). They are labeled as estimates in the card.
- Bits-per-weight numbers for the quant guide are standard approximations,
  not per-file measurements.
- GGUF v1/v2 headers parse, but real v1 files are rare and untested against
  real samples (synthetic files only).
- LM Studio .mflux / .safetensors formats are not parsed; only `.gguf` files
  in the LM Studio library are recognized (most local models are GGUF).
- Polish quality depends on the local model behind the endpoint; the prompt
  constrains style but can't guarantee taste. It's opt-in, off by default.
- No streaming/async polish; single request with a configurable timeout.
- Tested on Linux + Python 3.13 only; stdlib-only code should port fine to
  3.10+ on macOS/Windows (`~/.lmstudio` path detection included),
  but untested there.
