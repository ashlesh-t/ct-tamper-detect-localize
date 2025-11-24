import logging
import os
from logging.handlers import RotatingFileHandler

# Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.getenv(
    "LOG_DIR",
    os.path.join(os.path.dirname(__file__), "..", "logs")
)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Formatter
fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
formatter = logging.Formatter(fmt)

# Root logger setup (idempotent)
root = logging.getLogger()
if not root.handlers:
    root.setLevel(LOG_LEVEL)

    fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5)
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(LOG_LEVEL)
    sh.setFormatter(formatter)
    root.addHandler(sh)


def get_logger(name: str = __name__) -> logging.Logger:
    """Return a configured logger."""
    return logging.getLogger(name)