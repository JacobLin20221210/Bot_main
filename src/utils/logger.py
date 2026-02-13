import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path


def get_git_commit_hash():
    """Get the current git commit hash (short)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def setup_logger(name: str = "bot_or_not") -> logging.Logger:
    """
    Setup logger that writes to ./logs/{commit-hash}/{datetime}.log
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    # Generate log path
    commit_hash = get_git_commit_hash()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create logs directory structure: ./logs/{commit-hash}/{datetime}.log
    logs_dir = Path("./logs") / commit_hash
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    log_filename = f"{timestamp}.log"
    log_path = logs_dir / log_filename
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler - detailed logging
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
    )
    file_handler.setFormatter(file_format)
    
    # Console handler - info level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_format)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to: {log_path}")
    logger.info(f"Git commit: {commit_hash}")
    
    return logger


# Global logger instance
_logger = None


def get_logger() -> logging.Logger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


if __name__ == "__main__":
    # Test the logger
    logger = get_logger()
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.info("Logger test completed successfully!")
