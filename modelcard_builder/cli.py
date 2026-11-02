"""Command-line interface for modelcard-builder."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .card import render_card, card_filename
from .config import Config, load_config
from .gguf import GgufError, ModelInfo, human_size, read_gguf
from .polish import PolishError, fact_sheet, polish_text
from .sources import SourceError, lmstudio_dirs, lmstudio_models, ollama_list, ollama_model_info

VERSION = "0.1.0"


def _info_from_gguf(path: str) -> ModelInfo:
    return read_gguf(path)


def _emit(info: ModelInfo, args) -> int:
    polished = None
    if args.polish:
        cfg = Config(
            endpoint=args.endpoint or _config().endpoint,
            polish_model=args.model or _config().polish_model,
            timeout=args.timeout,
            api_key_env=args.api_key_env or _config().api_key_env,
        )
        try:
            polished = polish_text(fact_sheet(info), cfg)
        except PolishError as exc:
            print(f"note: polish skipped ({exc}); card written without it", file=sys.stderr)
    text = render_card(info, polished=polished)
    out = Path(args.out) if args.out else Path(card_filename(info))
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


_cfg_cache = None


def _config():
    global _cfg_cache
    if _cfg_cache is None:
        _cfg_cache = load_config()
    return _cfg_cache


def _add_polish_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--polish", action="store_true",
                   help="ask a local OpenAI-compatible model to add an 'In a nutshell' section")
    p.add_argument("--endpoint", help="OpenAI-compatible endpoint (default: Ollama's http://localhost:11434/v1)")
    p.add_argument("--model", help="model name for the polish step (default: llama3.2)")
    p.add_argument("--timeout", type=float, default=_DEFAULT_TIMEOUT,
                   help="seconds to wait for the polish model")
    p.add_argument("--api-key-env", help="name of the env var holding an optional API key (value never stored)")
    p.add_argument("-o", "--out", help="where to write the markdown card")


_DEFAULT_TIMEOUT = 120.0


def cmd_gguf(args) -> int:
    try:
        info = _info_from_gguf(args.file)
    except (GgufError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return _emit(info, args)


def cmd_ollama(args) -> int:
    if not args.model:
        try:
            models = ollama_list()
        except SourceError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if not models:
            print("error: `ollama list` shows no models installed", file=sys.stderr)
            return 1
        if len(models) == 1:
            args.model = models[0]
        else:
            print("Several models are installed; run with a name:", file=sys.stderr)
            print("  modelcard ollama <name>", file=sys.stderr)
            for mm in models:
                print(f"  - {mm}", file=sys.stderr)
            return 2
    try:
        info = ollama_model_info(args.model)
    except SourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return _emit(info, args)


def cmd_lmstudio(args) -> int:
    try:
        infos = lmstudio_models(args.dir)
    except SourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.list:
        if not infos:
            print("No GGUF files found in the LM Studio models directory.", file=sys.stderr)
            return 1
        print("GGUF files under LM Studio models:")
        for info in infos:
            print(f"  - {info.file_path.name}  ({human_size(info.file_size)})")
        return 0
    if not infos:
        print("No GGUF files found in the LM Studio models directory.", file=sys.stderr)
        return 1
    if args.model:
        matches = [i for i in infos if args.model.lower() in i.file_path.name.lower()]
        if not matches:
            print(f"no model matching '{args.model}'", file=sys.stderr)
            return 1
        info = matches[0]
    else:
        if len(infos) > 1:
            infos.sort(key=lambda i: i.file_path.stat().st_mtime, reverse=True)
            print(f"note: {len(infos)} GGUF files found; using the most recent:", file=sys.stderr)
        info = infos[0]
    return _emit(info, args)


def cmd_scan(args) -> int:
    print("Looking around your machine for local models...\n")
    anything = False

    try:
        models = ollama_list()
        print(f"Ollama ({len(models)} model(s)):")
        for mm in models:
            print(f"  - {mm}")
        if not models:
            print("  (none installed)")
        print()
        anything = True
    except SourceError as exc:
        print(f"Ollama: not found - {exc}\n")

    try:
        dirs_ready = lmstudio_dirs()
        if dirs_ready:
            for d in dirs_ready:
                infos = lmstudio_models(d)
                print(f"LM Studio ({d}) — {len(infos)} gguf file(s):")
                for info in infos:
                    print(f"  - {info.file_path.name}  {human_size(info.file_size)}")
            print()
            anything = True
        else:
            print("LM Studio: no models directory found (that's OK).\n")
    except (SourceError, OSError):
        print("LM Studio: could not scan models directory.\n")

    cwd_gguf = list(Path().glob("*.gguf")) + list(Path("models").glob("*.gguf"))
    if cwd_gguf:
        print("GGUF files in this folder:")
        for p in cwd_gguf:
            print(f"  - {p}  {human_size(p.stat().st_size)}")
        print()
        anything = True

    if not anything:
        print(
            "No local models found yet. Tip: `modelcard gguf path/to/model.gguf` "
            "works even before any server app is installed."
        )
    return 0


def cmd_config(args) -> int:
    cfg = _config()
    if args.init:
        target = Path(args.init)
        if target.exists():
            print(f"error: {target} already exists", file=sys.stderr)
            return 1
        target.write_text(default_config_toml(), encoding="utf-8")
        print(f"wrote {target} — edit to point the polish step anywhere OpenAI-compatible")
        if target != Path("modelcard.toml"):
            print("note: modelcard.toml in the working directory is read first")
        return 0
    print("polish endpoint :", cfg.endpoint)
    print("polish model    :", cfg.polish_model)
    print("timeout         :", f"{cfg.timeout:.0f}s")
    print("api key env var :", cfg.api_key_env, "(value is never stored or printed)")
    return 0


def default_config_toml() -> str:
    return """# modelcard-builder polish settings (all optional)
# Endpoint of any OpenAI-compatible chat server. Default is local Ollama.
[polish]
endpoint = "http://localhost:11434/v1"
model = "llama3.2"
timeout = 120
# If your endpoint needs a key, its env var NAME goes here; the value is only
# ever read from the environment, never saved in this file.
api_key_env = "MC_API_KEY"
"""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="modelcard",
        description=(
            "Generate a friendly markdown model card from local LLM files: "
            "GGUF files, Ollama models, or LM Studio installs."
        ),
    )
    ap.add_argument("--version", action="version", version=f"modelcard-builder {VERSION}")
    sub = ap.add_subparsers(dest="command", metavar="COMMAND")

    sub_gguf = sub.add_parser("gguf", help="build a card from a .gguf file")
    sub_gguf.add_argument("file", help="path to a .gguf file")
    _add_polish_flags(sub_gguf)
    sub_gguf.set_defaults(func=cmd_gguf)

    sub_ollama = sub.add_parser("ollama", help="build a card from an Ollama model")
    sub_ollama.add_argument("model", nargs="?", help="model name as shown by `ollama list`")
    _add_polish_flags(sub_ollama)
    sub_ollama.set_defaults(func=cmd_ollama)

    sub_lms = sub.add_parser("lmstudio", help="build a card from a LM Studio-installed model")
    sub_lms.add_argument("--dir", help="LM Studio models directory (default: auto-detect)")
    sub_lms.add_argument("model", nargs="?", default=None,
                         help="a few letters to pick the file, e.g. 'llama-8b-q4'")
    sub_lms.add_argument("--list", action="store_true", help="just list what was found")
    _add_polish_flags(sub_lms)
    sub_lms.set_defaults(func=cmd_lmstudio)

    sub_scan = sub.add_parser("scan", help="look around for local models")
    sub_scan.set_defaults(func=cmd_scan)

    sub_cfg = sub.add_parser("config", help="show polish settings, or write a starter modelcard.toml")
    sub_cfg.add_argument("--init", nargs="?", const="modelcard.toml", metavar="FILE",
                         help="write a starter config file (default: ./modelcard.toml)")
    sub_cfg.set_defaults(func=cmd_config)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "command", None):
        ap.print_help()
        return 0
    if not hasattr(args, "timeout"):
        args.timeout = _DEFAULT_TIMEOUT
    if not hasattr(args, "polish"):
        args.polish = False
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
