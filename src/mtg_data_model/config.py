from __future__ import annotations

from dataclasses import dataclass
import os


def _env(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val is not None and val != "" else default


@dataclass
class Settings:
    data_dir: str = _env("MTG_DATA_DIR", "data")
    log_dir: str = _env("MTG_LOG_DIR", "log")


settings = Settings()
