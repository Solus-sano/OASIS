from __future__ import annotations
import logging
import os
from typing import Optional

try:
    from ..config import OasisConfig
except Exception:
    OasisConfig = object

DEFAULT_CONSOLE_FMT = "[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
DEFAULT_FILE_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

ANSI_RESET = "\x1b[0m"
ANSI_COLOR_BY_LEVEL = {
    logging.DEBUG: "\x1b[90m",
    logging.INFO: "\x1b[36m",
    logging.WARNING: "\x1b[33m",
    logging.ERROR: "\x1b[31m",
    logging.CRITICAL: "\x1b[1;41;97m",
}


class ColorFormatter(logging.Formatter):
    def __init__(self, fmt: str, datefmt: Optional[str] = None):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        color_prefix = ANSI_COLOR_BY_LEVEL.get(record.levelno)
        if color_prefix:
            try:
                record.levelname = f"{color_prefix}{original_levelname}{ANSI_RESET}"
                return super().format(record)
            finally:
                record.levelname = original_levelname
        return super().format(record)


def _make_console_formatter(enable_color: bool) -> logging.Formatter:
    if enable_color:
        return ColorFormatter(DEFAULT_CONSOLE_FMT, datefmt="%H:%M:%S")
    return logging.Formatter(DEFAULT_CONSOLE_FMT, datefmt="%H:%M:%S")


def _ensure_root_console_handler(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    is_tty = getattr(console_handler.stream, "isatty", lambda: False)()
    force_color = bool(os.environ.get("FORCE_COLOR"))
    no_color = bool(os.environ.get("NO_COLOR"))
    color_enabled = (is_tty or force_color) and not no_color
    console_handler.setFormatter(_make_console_formatter(color_enabled))
    root.addHandler(console_handler)
    root.setLevel(level)


def get_logger(name: str = "oasis"):
    _ensure_root_console_handler()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def setup_logging_from_config(
    cfg: OasisConfig,
    *,
    name: str = "oasis",
    out_dir: Optional[str] = None,
    filename: Optional[str] = None,
    level: int = logging.INFO,
    overwrite: bool = True,
    color_console: bool = True,
) -> str:
    out_dir = out_dir or getattr(getattr(cfg, "event_forest", cfg), "out_dir", "output")
    os.makedirs(out_dir, exist_ok=True)
    log_filename = filename or f"{name}.log"
    log_path = os.path.join(out_dir, log_filename)

    root = logging.getLogger()
    while root.handlers:
        root.handlers.pop().close()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    is_tty = getattr(console_handler.stream, "isatty", lambda: False)()
    force_color = bool(os.environ.get("FORCE_COLOR"))
    no_color = bool(os.environ.get("NO_COLOR"))
    color_enabled = color_console and (is_tty or force_color) and not no_color
    console_handler.setFormatter(_make_console_formatter(color_enabled))

    file_mode = "w" if overwrite else "a"
    file_handler = logging.FileHandler(log_path, mode=file_mode, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(DEFAULT_FILE_FMT, datefmt="%Y-%m-%d %H:%M:%S"))

    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root.setLevel(level)

    return log_path
