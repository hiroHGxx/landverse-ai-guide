"""
Enhanced logging system for RAG application
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from ..config import get_config


class RAGLogger:
    """Enhanced logger for RAG system"""
    
    _loggers = {}
    _initialized = False
    
    @classmethod
    def setup_logging(cls, config=None) -> None:
        """Setup logging configuration"""
        if cls._initialized:
            return
        
        if config is None:
            config = get_config().logging
        
        # Create logs directory
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(config.format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler with rotation
        log_file = log_dir / config.log_file
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=cls._parse_size(config.max_file_size),
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, config.level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get logger instance"""
        if not cls._initialized:
            cls.setup_logging()
        
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        
        return cls._loggers[name]
    
    @staticmethod
    def _parse_size(size_str: str) -> int:
        """Parse size string to bytes"""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)


def get_logger(name: str) -> logging.Logger:
    """Convenience function to get logger"""
    return RAGLogger.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capability to any class"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return get_logger(self.__class__.__module__ + '.' + self.__class__.__name__)


# Decorator for function logging
def log_function_call(logger_name: Optional[str] = None):
    """Decorator to log function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                logger.info(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed with error: {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator


# Exception handling decorator
def handle_exceptions(logger_name: Optional[str] = None, reraise: bool = True):
    """Decorator to handle and log exceptions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {e}", exc_info=True)
                if reraise:
                    raise
                return None
        
        return wrapper
    return decorator