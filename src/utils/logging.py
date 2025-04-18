import logging
import os
from datetime import datetime
from typing import Optional, Union, Dict, Any

class Logger:
    """
    Custom logger for statue detection project.
    Handles both file and console logging with proper formatting.
    """
    def __init__(
        self, 
        name: str, 
        log_dir: str = "logs", 
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
    ) -> None:
        os.makedirs(log_dir, exist_ok=True)
        
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        log_file = os.path.join(log_dir, f"{name}.log")
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        # Clear existing handlers if any
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        # self.logger.addHandler(console_handler)
    
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.exception(msg, *args, **kwargs)

def get_logger(
    name: str, 
    log_dir: str = "logs", 
    console_level: int = logging.INFO, 
    file_level: int = logging.DEBUG
) -> Logger:
    """Get logger instance with specified configuration"""
    return Logger(name, log_dir, console_level, file_level)