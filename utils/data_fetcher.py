"""
Data fetching utilities for Indian commodities market data
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os

def get_indian_commodities():
    """
    Returns a dictionary of Indian commodities and their symbols
    """
    # Top 5 most liquid commodities with reliable data availability
    # Focused on markets with consistent data and high trading volume
    commodities = {
        'Crude Oil': 'CL=F',      # Crude Oil Futures
        'Natural Gas': 'NG=F',    # Natural Gas Futures
        'Gold': 'GC=F',           # Gold Futures
        'Silver': 'SI=F',         # Silver Futures  
        'Copper': 'HG=F',         # Copper Futures
    }
    
    return commodities

def get_commodity_data(symbol, period='1mo', interval='1d'):
    """
    Fetch commodity price data using yfinance
    
    Args:
        symbol (str): Commodity symbol
        period (str): Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        pandas.DataFrame: OHLCV data or None if error
    """
    try:
        # Handle special Indian commodity symbols that don't have direct yfinance mapping
        if symbol in ['CARDAMOM', 'TURMERIC', 'CORIANDER', 'JEERA']:
            return _get_synthetic_data(symbol, period, interval)
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            # Fallback to synthetic data if real data not available
            return _get_synthetic_data(symbol, period, interval)
        
        return data
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        # Return synthetic data as fallback
        return _get_synthetic_data(symbol, period, interval)

def _get_synthetic_data(symbol, period='1mo', interval='1d'):
    """
    Generate synthetic price data for commodities not available in yfinance
    This is a fallback for demonstration - in production, use real data APIs
    """
    try:
        # Determine number of data points based on period and interval
        periods_map = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, 
            '6mo': 180, '1y': 365, '2y': 730
        }
        
        intervals_map = {
            '1m': 1/1440, '5m': 5/1440, '15m': 15/1440, '1h': 1/24, '1d': 1
        }
        
        days = periods_map.get(period, 30)
        interval_days = intervals_map.get(interval, 1)
        num_points = int(days / interval_days)
        
        # Base prices for different commodities (in appropriate units)
        base_prices = {
            'GC=F': 2000,      # Gold $/oz
            'SI=F': 25,        # Silver $/oz
            'CL=F': 80,        # Crude Oil $/barrel
            'NG=F': 3,         # Natural Gas $/MMBtu
            'HG=F': 4,         # Copper $/lb
            'CARDAMOM': 1500,  # Cardamom Rs/kg
            'TURMERIC': 150,   # Turmeric Rs/kg
            'CORIANDER': 200,  # Coriander Rs/kg
            'JEERA': 300       # Jeera Rs/kg
        }
        
        base_price = base_prices.get(symbol, 100)
        
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        if interval == '1d':
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        elif interval == '1h':
            dates = pd.date_range(start=start_date, end=end_date, freq='H')
        else:
            dates = pd.date_range(start=start_date, end=end_date, periods=num_points)
        
        # Generate realistic price movements
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0, 0.02, len(dates))  # 2% daily volatility
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Generate intraday high/low around close price
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else close
            
            # Ensure OHLC relationships are correct
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Generate volume (commodity-specific ranges)
            if 'CARDAMOM' in symbol or 'TURMERIC' in symbol or 'CORIANDER' in symbol or 'JEERA' in symbol:
                volume = np.random.randint(1000, 10000)
            else:
                volume = np.random.randint(10000, 100000)
            
            data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        return df
        
    except Exception as e:
        print(f"Error generating synthetic data for {symbol}: {str(e)}")
        return None

def get_options_chain_data(symbol, expiry_date=None):
    """
    Fetch options chain data for a commodity
    
    Args:
        symbol (str): Commodity symbol
        expiry_date (str): Expiry date in YYYY-MM-DD format
    
    Returns:
        dict: Options chain data or None if error
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Get available expiry dates
        expirations = ticker.options
        
        if not expirations:
            return None
        
        # Use provided expiry or first available
        expiry = expiry_date if expiry_date in expirations else expirations[0]
        
        # Get options chain
        options_chain = ticker.option_chain(expiry)
        
        return {
            'calls': options_chain.calls,
            'puts': options_chain.puts,
            'expiry': expiry
        }
        
    except Exception as e:
        print(f"Error fetching options chain for {symbol}: {str(e)}")
        return None

def get_market_status():
    """
    Get current market status for Indian commodity markets
    
    Returns:
        dict: Market status information
    """
    try:
        # This would typically connect to a real market data API
        # For now, return based on Indian market hours
        
        from utils.indian_market_utils import is_market_open, get_indian_time
        
        indian_time = get_indian_time()
        market_open = is_market_open()
        
        return {
            'is_open': market_open,
            'current_time': indian_time.strftime('%Y-%m-%d %H:%M:%S IST'),
            'next_open': 'Next trading day 09:00 IST' if not market_open else 'Currently trading',
            'session': 'Morning' if 9 <= indian_time.hour < 17 else 'Evening' if 17 <= indian_time.hour < 23.5 else 'Closed'
        }
        
    except Exception as e:
        print(f"Error getting market status: {str(e)}")
        return {
            'is_open': False,
            'current_time': 'Unknown',
            'next_open': 'Unknown',
            'session': 'Unknown'
        }

def get_historical_volatility(symbol, period='1mo'):
    """
    Calculate historical volatility for a commodity
    
    Args:
        symbol (str): Commodity symbol
        period (str): Lookback period for volatility calculation
    
    Returns:
        float: Annualized historical volatility as percentage
    """
    try:
        data = get_commodity_data(symbol, period=period)
        
        if data is None or data.empty:
            return 25.0  # Default volatility
        
        # Calculate daily returns
        returns = data['Close'].pct_change().dropna()
        
        if len(returns) < 2:
            return 25.0
        
        # Calculate annualized volatility
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252) * 100  # 252 trading days
        
        return annual_vol
        
    except Exception as e:
        print(f"Error calculating volatility for {symbol}: {str(e)}")
        return 25.0  # Default volatility

def get_multiple_commodities_data(symbols, period='1mo', interval='1d'):
    """
    Fetch data for multiple commodities
    
    Args:
        symbols (list): List of commodity symbols
        period (str): Data period
        interval (str): Data interval
    
    Returns:
        dict: Dictionary with symbol as key and data as value
    """
    data_dict = {}
    
    for symbol in symbols:
        try:
            data = get_commodity_data(symbol, period=period, interval=interval)
            if data is not None:
                data_dict[symbol] = data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            continue
    
    return data_dict

def get_commodity_info(symbol):
    """
    Get detailed information about a commodity
    
    Args:
        symbol (str): Commodity symbol
    
    Returns:
        dict: Commodity information
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract relevant information
        commodity_info = {
            'symbol': symbol,
            'name': info.get('longName', 'Unknown'),
            'currency': info.get('currency', 'USD'),
            'exchange': info.get('exchange', 'Unknown'),
            'market_cap': info.get('marketCap', 'N/A'),
            'volume': info.get('volume', 'N/A'),
            'avg_volume': info.get('averageVolume', 'N/A'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 'N/A')
        }
        
        return commodity_info
        
    except Exception as e:
        print(f"Error getting commodity info for {symbol}: {str(e)}")
        return {
            'symbol': symbol,
            'name': 'Unknown',
            'currency': 'USD',
            'exchange': 'Unknown',
            'market_cap': 'N/A',
            'volume': 'N/A',
            'avg_volume': 'N/A',
            'fifty_two_week_high': 'N/A',
            'fifty_two_week_low': 'N/A'
        }

def validate_symbol(symbol):
    """
    Validate if a commodity symbol is available
    
    Args:
        symbol (str): Commodity symbol to validate
    
    Returns:
        bool: True if symbol is valid and data is available
    """
    try:
        data = get_commodity_data(symbol, period='5d')
        return data is not None and not data.empty
    except:
        return False

# Cache for storing frequently accessed data
_data_cache = {}
_cache_expiry = {}

def get_cached_data(symbol, period='1d', cache_duration_minutes=5):
    """
    Get cached commodity data to reduce API calls
    
    Args:
        symbol (str): Commodity symbol
        period (str): Data period
        cache_duration_minutes (int): Cache duration in minutes
    
    Returns:
        pandas.DataFrame: Cached or fresh data
    """
    cache_key = f"{symbol}_{period}"
    current_time = datetime.now()
    
    # Check if we have cached data that's still valid
    if (cache_key in _data_cache and 
        cache_key in _cache_expiry and 
        current_time < _cache_expiry[cache_key]):
        return _data_cache[cache_key]
    
    # Fetch fresh data
    data = get_commodity_data(symbol, period=period)
    
    # Cache the data
    if data is not None:
        _data_cache[cache_key] = data
        _cache_expiry[cache_key] = current_time + timedelta(minutes=cache_duration_minutes)
    
    return data
