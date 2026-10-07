"""
Logger configuration module.

Set LOG_LEVEL=DEBUG in .env to see retrieved context and intermediate results.
"""

import logging
import os

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)


logger = get_logger("adaptive_rag")
