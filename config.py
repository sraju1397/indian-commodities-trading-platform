"""
Configuration management for the trading platform
"""
import os
from utils.logger import logger

class Config:
    """Configuration settings for the application"""
    
    # Kite Connect API Configuration
    KITE_API_KEY = os.getenv('KITE_API_KEY', '')
    KITE_API_SECRET = os.getenv('KITE_API_SECRET', '')
    KITE_ACCESS_TOKEN = os.getenv('KITE_ACCESS_TOKEN', '')
    
    # Market Data Configuration
    DEFAULT_TIMEFRAME = '1d'
    DEFAULT_INTERVAL = '5m'
    AUTO_REFRESH_INTERVAL = 30  # seconds
    
    # Cache Configuration
    CACHE_TTL = 300  # 5 minutes
    
    # Trading Configuration
    TRADING_ENABLED = os.getenv('TRADING_ENABLED', 'false').lower() == 'true'
    PAPER_TRADING = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
    
    # Risk Management
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '100000'))  # Maximum position size in INR
    STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', '2.0'))    # Default stop loss percentage
    
    # UI Configuration
    THEME = {
        'PRIMARY_COLOR': '#1a1a1a',
        'SECONDARY_COLOR': '#28a745',
        'ERROR_COLOR': '#dc3545',
        'WARNING_COLOR': '#ffc107',
        'INFO_COLOR': '#17a2b8'
    }
    
    @classmethod
    def validate(cls):
        """
        Validate configuration settings
        
        Returns:
            tuple: (bool, list) - (is_valid, list of validation messages)
        """
        messages = []
        is_valid = True
        
        # Check API credentials
        if not cls.KITE_API_KEY:
            messages.append("Warning: Kite API key not set")
            is_valid = False
        
        if not cls.KITE_API_SECRET:
            messages.append("Warning: Kite API secret not set")
            is_valid = False
            
        # Check trading configuration
        if cls.TRADING_ENABLED and not cls.KITE_ACCESS_TOKEN:
            messages.append("Error: Trading enabled but access token not set")
            is_valid = False
            
        # Validate risk parameters
        if cls.MAX_POSITION_SIZE <= 0:
            messages.append("Error: Invalid maximum position size")
            is_valid = False
            
        if not (0 < cls.STOP_LOSS_PERCENT < 100):
            messages.append("Error: Invalid stop loss percentage")
            is_valid = False
        
        return is_valid, messages
    
    @classmethod
    def log_config(cls):
        """Log current configuration settings"""
        logger.info("Current Configuration:")
        logger.info(f"Trading Enabled: {cls.TRADING_ENABLED}")
        logger.info(f"Paper Trading: {cls.PAPER_TRADING}")
        logger.info(f"Max Position Size: ₹{cls.MAX_POSITION_SIZE:,.2f}")
        logger.info(f"Stop Loss: {cls.STOP_LOSS_PERCENT}%")
        
        # Log validation results
        is_valid, messages = cls.validate()
        if not is_valid:
            for message in messages:
                logger.warning(message)
    
    @classmethod
    def initialize(cls):
        """Initialize and validate configuration"""
        is_valid, messages = cls.validate()
        
        if not is_valid:
            logger.warning("Configuration validation failed:")
            for message in messages:
                logger.warning(message)
        
        cls.log_config()
        return is_valid

# Initialize configuration when module is imported
Config.initialize()
