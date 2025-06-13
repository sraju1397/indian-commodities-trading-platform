"""
Indian market utilities for timing and status
"""
import pytz
from datetime import datetime, time, timedelta

def get_indian_time():
    """
    Get current Indian Standard Time (IST)
    
    Returns:
        datetime: Current IST time
    """
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist)

def is_market_open():
    """
    Check if Indian commodity markets are currently open
    
    Returns:
        bool: True if markets are open, False otherwise
    """
    current_time = get_indian_time()
    current_weekday = current_time.weekday()  # Monday = 0, Sunday = 6
    
    # Markets are closed on weekends
    if current_weekday >= 5:  # Saturday (5) and Sunday (6)
        return False
    
    # Market hours (IST):
    # Morning session: 9:00 AM to 5:00 PM
    # Evening session: 5:00 PM to 11:30 PM
    current_time_only = current_time.time()
    
    morning_start = time(9, 0)   # 9:00 AM
    evening_end = time(23, 30)   # 11:30 PM
    
    # Check if current time falls within trading hours
    return morning_start <= current_time_only <= evening_end

def get_market_status():
    """
    Get detailed market status string
    
    Returns:
        str: Market status description
    """
    if is_market_open():
        current_time = get_indian_time()
        current_hour = current_time.hour
        
        if 9 <= current_hour < 17:
            return "OPEN - Morning Session"
        else:
            return "OPEN - Evening Session"
    else:
        current_time = get_indian_time()
        current_weekday = current_time.weekday()
        
        if current_weekday >= 5:
            return "CLOSED - Weekend"
        else:
            current_hour = current_time.hour
            if current_hour < 9:
                return "CLOSED - Pre-Market"
            else:
                return "CLOSED - Post-Market"

def get_next_market_open():
    """
    Get the next market opening time
    
    Returns:
        datetime: Next market opening time in IST
    """
    current_time = get_indian_time()
    current_weekday = current_time.weekday()
    
    # If it's weekend, next opening is Monday 9 AM
    if current_weekday >= 5:  # Saturday or Sunday
        days_until_monday = (7 - current_weekday) % 7
        if days_until_monday == 0:  # If it's Sunday
            days_until_monday = 1
        
        next_open = current_time.replace(hour=9, minute=0, second=0, microsecond=0)
        next_open = next_open + timedelta(days=days_until_monday)
        return next_open
    
    # If it's a weekday
    current_time_only = current_time.time()
    
    if current_time_only < time(9, 0):
        # Before market opens today
        return current_time.replace(hour=9, minute=0, second=0, microsecond=0)
    else:
        # Market already opened today or closed, next opening is tomorrow 9 AM
        if current_weekday == 4:  # Friday
            # Next opening is Monday
            next_open = current_time.replace(hour=9, minute=0, second=0, microsecond=0)
            next_open = next_open + timedelta(days=3)
            return next_open
        else:
            # Next opening is tomorrow
            next_open = current_time.replace(hour=9, minute=0, second=0, microsecond=0)
            next_open = next_open + timedelta(days=1)
            return next_open

def get_market_hours_info():
    """
    Get comprehensive market hours information
    
    Returns:
        dict: Market hours and status information
    """
    return {
        'morning_session': '09:00 - 17:00 IST',
        'evening_session': '17:00 - 23:30 IST',
        'trading_days': 'Monday to Friday',
        'current_status': get_market_status(),
        'is_open': is_market_open(),
        'current_time': get_indian_time().strftime('%Y-%m-%d %H:%M:%S IST'),
        'next_open': get_next_market_open().strftime('%Y-%m-%d %H:%M:%S IST')
    }

def get_session_info():
    """
    Get current session information
    
    Returns:
        dict: Current session details
    """
    current_time = get_indian_time()
    current_hour = current_time.hour
    
    if not is_market_open():
        return {
            'session': 'Closed',
            'session_type': 'No Trading',
            'time_remaining': 'Market Closed'
        }
    
    if 9 <= current_hour < 17:
        # Morning session
        session_end = current_time.replace(hour=17, minute=0, second=0, microsecond=0)
        time_remaining = session_end - current_time
        
        hours, remainder = divmod(time_remaining.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        return {
            'session': 'Morning Session',
            'session_type': 'Regular Trading',
            'time_remaining': f"{hours}h {minutes}m remaining"
        }
    else:
        # Evening session
        session_end = current_time.replace(hour=23, minute=30, second=0, microsecond=0)
        time_remaining = session_end - current_time
        
        hours, remainder = divmod(time_remaining.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        return {
            'session': 'Evening Session',
            'session_type': 'Extended Trading',
            'time_remaining': f"{hours}h {minutes}m remaining"
        }

def is_trading_holiday(date=None):
    """
    Check if a given date is a trading holiday
    Note: This is a simplified version. In production, you would
    integrate with a holiday calendar API.
    
    Args:
        date (datetime, optional): Date to check. Defaults to today.
    
    Returns:
        bool: True if it's a trading holiday
    """
    if date is None:
        date = get_indian_time().date()
    
    # Basic weekend check
    if date.weekday() >= 5:
        return True
    
    # Add major Indian holidays here
    # This would typically come from an API or database
    major_holidays_2024 = [
        # Add actual holiday dates as needed
        # Format: (month, day)
        (1, 26),   # Republic Day
        (8, 15),   # Independence Day
        (10, 2),   # Gandhi Jayanti
    ]
    
    for month, day in major_holidays_2024:
        if date.month == month and date.day == day:
            return True
    
    return False

def get_time_until_market_open():
    """
    Get time remaining until next market opening
    
    Returns:
        dict: Time breakdown until market opens
    """
    if is_market_open():
        return {
            'is_open': True,
            'message': 'Market is currently open'
        }
    
    next_open = get_next_market_open()
    current_time = get_indian_time()
    time_diff = next_open - current_time
    
    days = time_diff.days
    hours, remainder = divmod(time_diff.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    
    return {
        'is_open': False,
        'days': days,
        'hours': hours,
        'minutes': minutes,
        'total_minutes': int(time_diff.total_seconds() / 60),
        'next_open_time': next_open.strftime('%Y-%m-%d %H:%M:%S IST'),
        'message': f"Market opens in {days}d {hours}h {minutes}m"
    }