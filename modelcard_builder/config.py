"""Configuration defaults for the optional LLM polish step.

Priority: CLI flags > environment variables (MC_ENDPOINT, MC_POLISH_MODEL) >
modelcard.toml (cwd, then ~/.config/modelcard-builder/config.toml) > defaults.
No API keys are ever written to disk; keys are only read from the env var named
by MC_API_KEY_ENV (or --api-key-env) when a endpoint requires one.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ENDPOINT = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.2"
DEFAULT_TIMEOUT = 120.0


@dataclass
class Config:
    endpoint: str = DEFAULT_ENDPOINT
    polish_model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    api_key_env: str = "MC_API_KEY"

    @property
    def endpoint_root(self) -> str:
        e = self.endpoint.rstrip("/")
        return e[:-3] if e.endswith("/v1") else e

    @property
    def chat_url(self) -> str:
        return f"{self.endpoint_root}/v1/chat/completions"


def _candidate_files() -> list[Path]:
    cfg_dir = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config"
    )
    return [
        Path("modelcard.toml"),
        Path(cfg_dir) / "modelcard-builder" / "config.toml",
    ]


def load_config() -> Config:
    cfg = Config()
    for candidate in _candidate_files():
        try:
            with open(candidate, "rb") as fh:
                data = tomllib.load(fh)
        except FileNotFoundError:
            continue
        except tomllib.TOMLDecodeError:
            continue
        section = data.get("polish", {})
        if "endpoint" in section:
            cfg.endpoint = str(section["endpoint"])
        if "model" in section:
            cfg.polish_model = str(section["model"])
        if "timeout" in section:
            cfg.timeout = float(section["timeout"])
        if "api_key_env" in section:
            cfg.api_key_env = str(section["api_key_env"])
        break

    if os.environ.get("MC_ENDPOINT"):
        cfg.endpoint = os.environ["MC_ENDPOINT"]
    if os.environ.get("MC_POLISH_MODEL"):
        cfg.polish_model = os.environ["MC_POLISH_MODEL"]
    if os.environ.get("MC_API_KEY"):
        cfg.api_key_env = "MC_API_KEY"
    return cfg
