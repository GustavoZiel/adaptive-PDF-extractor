"""Custom logger with colored output and global log level setting."""

import logging
import re
import sys

# Global log level that can be set before creating loggers
_GLOBAL_LOG_LEVEL = logging.INFO


def set_global_log_level(level):
    """Set the global log level for all loggers created afterwards.

    Args:
        level: Logging level (logging.DEBUG, logging.INFO, etc.)
    """
    global _GLOBAL_LOG_LEVEL
    _GLOBAL_LOG_LEVEL = level

    # Update all existing loggers
    for logger_name in logging.Logger.manager.loggerDict:
        existing_logger = logging.getLogger(logger_name)
        if existing_logger.handlers:
            existing_logger.setLevel(level)
            for handler in existing_logger.handlers:
                handler.setLevel(level)


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages. (Inspired on Karpathy's nanochat code hehe)"""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
            )
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == "INFO":
            # Highlight numbers and percentages
            message = re.sub(
                r"(\d+\.?\d*\s*(?:GB|MB|%|docs))",
                rf"{self.BOLD}\1{self.RESET}",
                message,
            )
            message = re.sub(
                r"(Shard \d+)",
                rf"{self.COLORS['INFO']}{self.BOLD}\1{self.RESET}",
                message,
            )
        return message


def get_logger(name, level=None, log_file=None):
    """Creates and returns a logger.

    Args:
        name (str): Logger name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, uses the global log level.
        log_file (str, optional): Path to log file. If None, only prints to console.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Use global log level if no level specified
    if level is None:
        level = _GLOBAL_LOG_LEVEL

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate logs if root logger is used

    # Clear previous handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(
        ColoredFormatter(
            "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s ",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(ch)

    return logger
