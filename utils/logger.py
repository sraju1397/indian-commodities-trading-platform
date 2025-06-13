"""
Centralized logging configuration for the application
"""
import logging
import os
from datetime import datetime

def setup_logger(name='trading_platform'):
    """
    Configure and return a logger instance with both file and console handlers
    
    Args:
        name (str): Name of the logger instance
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    
    return logger

# Create global logger instance
logger = setup_logger()
