from typing import Any, Optional, Type, Union
from functools import wraps
import asyncio
import traceback
import sys
from src.utils.logging import get_logger

logger = get_logger("exceptions")

class StatueDetectionError(Exception):
    """Base exception for all statue detection errors"""
    pass

class DataError(StatueDetectionError):
    """Exception raised for errors in the data processing pipeline"""
    pass

class ModelError(StatueDetectionError):
    """Exception raised for errors in the model implementation or loading"""
    pass

class TrainingError(StatueDetectionError):
    """Exception raised for errors during model training"""
    pass

class InferenceError(StatueDetectionError):
    """Exception raised for errors during model inference"""
    pass

def exception_handler(func):
    """
    Decorator for synchronous function exception handling
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with exception handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except StatueDetectionError as e:
            logger.error(f"{e.__class__.__name__}: {str(e)}")
            raise
        except Exception as e:
            logger.critical(
                f"Unexpected error in {func.__name__}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            raise StatueDetectionError(f"Unexpected error in {func.__name__}: {str(e)}") from e
    return wrapper

def async_exception_handler(func):
    """
    Decorator for asynchronous function exception handling
    
    Args:
        func: Async function to decorate
        
    Returns:
        Decorated async function with exception handling
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except StatueDetectionError as e:
            logger.error(f"{e.__class__.__name__}: {str(e)}")
            raise
        except Exception as e:
            logger.critical(
                f"Unexpected error in {func.__name__}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            raise StatueDetectionError(f"Unexpected error in {func.__name__}: {str(e)}") from e
    return wrapper

def setup_global_exception_handler():
    """
    Set up global exception handler for uncaught exceptions
    """
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Let keyboard interrupts pass through
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    sys.excepthook = handle_exception
    
    # For asyncio exceptions
    loop = asyncio.get_event_loop()
    
    def handle_async_exception(loop, context):
        exception = context.get("exception")
        if exception:
            msg = f"Uncaught asyncio exception: {str(exception)}"
            logger.critical(msg, exc_info=exception)
        else:
            msg = f"Asyncio error: {context['message']}"
            logger.critical(msg)
    
    loop.set_exception_handler(handle_async_exception)