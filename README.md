# modelcard-builder

A tiny, friendly command-line tool that reads the **facts already on your machine**
about a local AI model — a `.gguf` file, an Ollama model, or a LM Studio download —
and turns them into a **readable one-page "model card" in Markdown**.

No cloud, no accounts, no API keys required. Pure Python standard library,
Python 3.10+. Everything stays on your computer.

```text
modelcard gguf ~/.models/llama-8b-Q4_K_M.gguf
wrote llama_8b_card.md
```

## What goes in the card

- **At a glance** — file, size, architecture, context length, estimated parameters
- **Quantization, explained like a friend would** — what `Q4_K_M` or `IQ3_XS`
  actually means for you: quality vs. memory, in plain English, no condescension
- **Practical use-cases** — suggested starting points based on the model's size
  (laptop-daily-driver vs. workstation-class), honestly labeled as estimates
- **What you need to run it** — sorted memory estimate derived from the real file size
- **Prompt style notes** — whether the file ships a chat template, plus per-family tips
- **My notes** — a fill-in section for *your* use-cases and prompts that worked;
  since the card is plain Markdown it grows with you
- *(Optional)* an **"In a nutshell"** section written by a local model via polish (below)

## Install / run

```bash
# run straight from this folder
python3 -m modelcard_builder --help

# or, after `pip install .`
modelcard --help
```

### Build a card from a GGUF file

```bash
modelcard gguf path/to/model-Q4_K_M.gguf          # writes <name>_card.md
modelcard gguf model.gguf -o lm studio notes.md   # custom output file
```

### Build a card from an Ollama model

```bash
modelcard ollama llama3.2     # name as shown by `ollama list`
modelcard ollama              # if only one model is installed, just uses it
```

### Build a card from a LM Studio install

```bash
modelcard lmstudio --list                  # see the gguf files LM Studio has
modelcard lmstudio llama-8b-q4            # match a few letters of the filename
modelcard lmstudio --dir /path/to/models m
```

LM Studio's models folder is auto-detected (`~/.lmstudio/models` and common
alternatives), or set `LMSTUDIO_MODELS_DIR`.

### See everything at once

```bash
modelcard scan      # Ollama models + LM Studio library + gguf files near you
```

## Optional "polish" from a local model

You can ask a local LLM (or any **OpenAI-compatible** endpoint) to write a warm,
plain-English "In a nutshell" paragraph into the card. **The default is a local
Ollama server**, so polish uses your own machine by default; polish is skipped
gracefully (with a note) if the endpoint is unreachable.

```bash
modelcard gguf model.gguf --polish                 # polish via Ollama (localhost:11434)
modelcard gguf model.gguf --polish \
    --endpoint http://localhost:1234/v1 --model my-favorite
```

Defaults come from, in order: CLI flags → environment variables
(`MC_ENDPOINT`, `MC_POLISH_MODEL`, `MC_API_KEY`) → `modelcard.toml`
(current folder, then `~/.config/modelcard-builder/config.toml`).
Generate a starter config with:

```bash
modelcard config --init
```

**Privacy note:** polish sends only the fact sheet (name, size, quantization,
context size) to the endpoint you chose, and **only if you ask for `--polish`**
— never otherwise. If your endpoint needs a key, the *name* of its environment
variable is configured in a file; the key itself is only ever read from the
environment and is never stored or printed anywhere.

## Tested

```bash
python3 -m unittest discover -s tests
```

Each test writes real (tiny, synthetic) `.gguf` files, runs a stub
OpenAI-compatible HTTP server on localhost, and exercises the CLI end-to-end.
No network access outside your own machine; no API keys.

## Who this is for

Built the same way [Dylan Asdale](https://dylanasdale.github.io) explains things
in *The Massive AI Manual*: like a friend would — plain English, step by step,
never condescending.

Released under the MIT License.
**Copyright (c) 2026 Asdale Dynamics, LLC** — a small LLC focused on
beginner-friendly AI education, books, and practical local-first tools.
