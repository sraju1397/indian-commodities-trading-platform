"""
Portfolio management utilities for tracking options positions
"""
import json
import pandas as pd
from datetime import datetime, timedelta
import os
from utils.database_manager import db_manager

class PortfolioManager:
    """
    Manages options portfolio including positions, trades, and performance tracking
    """
    
    def __init__(self, user_id='default_user'):
        """
        Initialize portfolio manager with database support
        
        Args:
            user_id (str): User identifier for database operations
        """
        self.user_id = user_id
        self.positions = []
        self.trades_history = []
        
        # Initialize database
        try:
            db_manager.initialize_database()
            self.load_portfolio()
        except Exception as e:
            print(f"Database initialization error: {e}")
            # Fallback to file-based storage
            self.portfolio_file = 'portfolio_data.json'
            self.load_portfolio_from_file()
    
    def load_portfolio(self):
        """Load portfolio data from database"""
        try:
            self.positions = db_manager.get_portfolio_positions(self.user_id, status="All")
            self.trades_history = db_manager.get_recommendations_history(self.user_id)
        except Exception as e:
            print(f"Error loading portfolio from database: {str(e)}")
            self.positions = []
            self.trades_history = []
    
    def load_portfolio_from_file(self):
        """Fallback: Load portfolio data from file"""
        try:
            if hasattr(self, 'portfolio_file') and os.path.exists(self.portfolio_file):
                with open(self.portfolio_file, 'r') as f:
                    data = json.load(f)
                    self.positions = data.get('positions', [])
                    self.trades_history = data.get('trades_history', [])
        except Exception as e:
            print(f"Error loading portfolio from file: {str(e)}")
            self.positions = []
            self.trades_history = []
    
    def save_portfolio(self):
        """Save portfolio data to file"""
        try:
            data = {
                'positions': self.positions,
                'trades_history': self.trades_history,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.portfolio_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving portfolio: {str(e)}")
    
    def add_position(self, commodity, option_type, action, strike_price, quantity, 
                    premium, expiry_date, notes=""):
        """
        Add a new option position to the portfolio
        
        Args:
            commodity (str): Commodity name
            option_type (str): 'Call' or 'Put'
            action (str): 'Buy' or 'Sell'
            strike_price (float): Strike price
            quantity (int): Number of lots
            premium (float): Premium per unit
            expiry_date (datetime or str): Expiry date
            notes (str): Optional notes about the trade
        
        Returns:
            dict: Position details
        """
        try:
            # Convert expiry_date to string if it's a datetime object
            if isinstance(expiry_date, datetime):
                expiry_str = expiry_date.strftime('%Y-%m-%d')
            else:
                expiry_str = str(expiry_date)
            
            position = {
                'id': len(self.positions),
                'commodity': commodity,
                'option_type': option_type,
                'action': action,
                'strike_price': float(strike_price),
                'quantity': int(quantity),
                'premium': float(premium),
                'expiry_date': expiry_str,
                'trade_date': datetime.now().strftime('%Y-%m-%d'),
                'status': 'Active',
                'notes': notes,
                'entry_time': datetime.now().isoformat()
            }
            
            self.positions.append(position)
            
            # Add to trades history
            trade_record = position.copy()
            trade_record['trade_type'] = 'Open'
            self.trades_history.append(trade_record)
            
            self.save_portfolio()
            return position
            
        except Exception as e:
            print(f"Error adding position: {str(e)}")
            return None
    
    def close_position(self, position_id, closing_premium=None, notes=""):
        """
        Close an existing position
        
        Args:
            position_id (int): Position ID to close
            closing_premium (float): Premium received/paid on closing
            notes (str): Notes about closing the position
        
        Returns:
            bool: True if successfully closed
        """
        try:
            if 0 <= position_id < len(self.positions):
                position = self.positions[position_id]
                
                if position['status'] == 'Active':
                    position['status'] = 'Closed'
                    position['close_date'] = datetime.now().strftime('%Y-%m-%d')
                    position['close_time'] = datetime.now().isoformat()
                    
                    if closing_premium is not None:
                        position['closing_premium'] = float(closing_premium)
                    
                    if notes:
                        position['close_notes'] = notes
                    
                    # Add to trades history
                    close_record = position.copy()
                    close_record['trade_type'] = 'Close'
                    self.trades_history.append(close_record)
                    
                    self.save_portfolio()
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error closing position: {str(e)}")
            return False
    
    def modify_position(self, position_id, **kwargs):
        """
        Modify an existing position
        
        Args:
            position_id (int): Position ID to modify
            **kwargs: Fields to update
        
        Returns:
            bool: True if successfully modified
        """
        try:
            if 0 <= position_id < len(self.positions):
                position = self.positions[position_id]
                
                # Only allow modification of active positions
                if position['status'] == 'Active':
                    for key, value in kwargs.items():
                        if key in position:
                            position[key] = value
                    
                    position['last_modified'] = datetime.now().isoformat()
                    
                    self.save_portfolio()
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error modifying position: {str(e)}")
            return False
    
    def get_all_positions(self):
        """
        Get all positions in the portfolio
        
        Returns:
            list: List of all positions
        """
        return self.positions.copy()
    
    def get_active_positions(self):
        """
        Get only active positions
        
        Returns:
            list: List of active positions
        """
        return [pos for pos in self.positions if pos['status'] == 'Active']
    
    def get_closed_positions(self):
        """
        Get only closed positions
        
        Returns:
            list: List of closed positions
        """
        return [pos for pos in self.positions if pos['status'] == 'Closed']
    
    def get_positions_by_commodity(self, commodity):
        """
        Get positions for a specific commodity
        
        Args:
            commodity (str): Commodity name
        
        Returns:
            list: Positions for the specified commodity
        """
        return [pos for pos in self.positions if pos['commodity'] == commodity]
    
    def get_positions_expiring_soon(self, days=7):
        """
        Get positions expiring within specified days
        
        Args:
            days (int): Number of days to look ahead
        
        Returns:
            list: Positions expiring soon
        """
        try:
            cutoff_date = datetime.now() + timedelta(days=days)
            expiring_positions = []
            
            for position in self.get_active_positions():
                expiry_date = datetime.strptime(position['expiry_date'], '%Y-%m-%d')
                if expiry_date <= cutoff_date:
                    position['days_to_expiry'] = (expiry_date - datetime.now()).days
                    expiring_positions.append(position)
            
            return sorted(expiring_positions, key=lambda x: x['days_to_expiry'])
            
        except Exception as e:
            print(f"Error getting expiring positions: {str(e)}")
            return []
    
    def calculate_portfolio_value(self, current_prices):
        """
        Calculate current portfolio value based on current market prices
        
        Args:
            current_prices (dict): Dictionary with commodity prices
        
        Returns:
            dict: Portfolio value metrics
        """
        try:
            total_value = 0
            total_cost = 0
            total_pnl = 0
            position_count = 0
            
            for position in self.get_active_positions():
                commodity = position['commodity']
                
                if commodity in current_prices:
                    current_price = current_prices[commodity]
                    
                    # Calculate position value (simplified)
                    position_cost = position['premium'] * position['quantity']
                    
                    # This is a simplified calculation - in reality, you'd need
                    # current option prices based on current underlying price
                    if position['action'] == 'Buy':
                        position_value = position_cost  # Placeholder
                        position_pnl = 0  # Would calculate based on current option price
                    else:
                        position_value = position_cost
                        position_pnl = 0
                    
                    total_value += position_value
                    total_cost += position_cost
                    total_pnl += position_pnl
                    position_count += 1
            
            return {
                'total_value': total_value,
                'total_cost': total_cost,
                'total_pnl': total_pnl,
                'position_count': position_count,
                'return_pct': (total_pnl / total_cost * 100) if total_cost > 0 else 0
            }
            
        except Exception as e:
            print(f"Error calculating portfolio value: {str(e)}")
            return {
                'total_value': 0,
                'total_cost': 0,
                'total_pnl': 0,
                'position_count': 0,
                'return_pct': 0
            }
    
    def get_portfolio_summary(self):
        """
        Get portfolio summary statistics
        
        Returns:
            dict: Portfolio summary
        """
        try:
            all_positions = self.get_all_positions()
            active_positions = self.get_active_positions()
            closed_positions = self.get_closed_positions()
            
            # Commodity breakdown
            commodity_breakdown = {}
            for position in active_positions:
                commodity = position['commodity']
                if commodity not in commodity_breakdown:
                    commodity_breakdown[commodity] = {
                        'positions': 0,
                        'total_premium': 0,
                        'calls': 0,
                        'puts': 0
                    }
                
                commodity_breakdown[commodity]['positions'] += 1
                commodity_breakdown[commodity]['total_premium'] += position['premium'] * position['quantity']
                
                if position['option_type'] == 'Call':
                    commodity_breakdown[commodity]['calls'] += 1
                else:
                    commodity_breakdown[commodity]['puts'] += 1
            
            # Calculate total invested
            total_invested = sum([pos['premium'] * pos['quantity'] 
                                for pos in active_positions 
                                if pos['action'] == 'Buy'])
            
            total_received = sum([pos['premium'] * pos['quantity'] 
                                for pos in active_positions 
                                if pos['action'] == 'Sell'])
            
            net_invested = total_invested - total_received
            
            # Expiry analysis
            expiry_breakdown = {}
            for position in active_positions:
                expiry = position['expiry_date']
                if expiry not in expiry_breakdown:
                    expiry_breakdown[expiry] = 0
                expiry_breakdown[expiry] += 1
            
            return {
                'total_positions': len(all_positions),
                'active_positions': len(active_positions),
                'closed_positions': len(closed_positions),
                'commodity_breakdown': commodity_breakdown,
                'total_invested': total_invested,
                'total_received': total_received,
                'net_invested': net_invested,
                'expiry_breakdown': expiry_breakdown,
                'last_trade_date': max([pos['trade_date'] for pos in all_positions]) if all_positions else None
            }
            
        except Exception as e:
            print(f"Error getting portfolio summary: {str(e)}")
            return {}
    
    def get_trades_history(self, start_date=None, end_date=None, commodity=None):
        """
        Get trades history with optional filters
        
        Args:
            start_date (str): Start date filter (YYYY-MM-DD)
            end_date (str): End date filter (YYYY-MM-DD)
            commodity (str): Commodity filter
        
        Returns:
            list: Filtered trades history
        """
        try:
            filtered_trades = self.trades_history.copy()
            
            if start_date:
                filtered_trades = [t for t in filtered_trades 
                                 if t['trade_date'] >= start_date]
            
            if end_date:
                filtered_trades = [t for t in filtered_trades 
                                 if t['trade_date'] <= end_date]
            
            if commodity:
                filtered_trades = [t for t in filtered_trades 
                                 if t['commodity'] == commodity]
            
            return filtered_trades
            
        except Exception as e:
            print(f"Error getting trades history: {str(e)}")
            return []
    
    def export_portfolio_to_csv(self, filename=None):
        """
        Export portfolio data to CSV
        
        Args:
            filename (str): Output filename
        
        Returns:
            str: Filename of exported file
        """
        try:
            if filename is None:
                filename = f"portfolio_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            df = pd.DataFrame(self.positions)
            df.to_csv(filename, index=False)
            
            return filename
            
        except Exception as e:
            print(f"Error exporting portfolio: {str(e)}")
            return None
    
    def import_portfolio_from_csv(self, filename):
        """
        Import portfolio data from CSV
        
        Args:
            filename (str): Input filename
        
        Returns:
            bool: True if successful
        """
        try:
            df = pd.read_csv(filename)
            
            # Validate required columns
            required_columns = ['commodity', 'option_type', 'action', 'strike_price', 
                              'quantity', 'premium', 'expiry_date']
            
            if not all(col in df.columns for col in required_columns):
                print("CSV file missing required columns")
                return False
            
            # Convert to list of dictionaries
            imported_positions = df.to_dict('records')
            
            # Add imported positions
            for pos in imported_positions:
                # Ensure proper data types
                pos['strike_price'] = float(pos['strike_price'])
                pos['quantity'] = int(pos['quantity'])
                pos['premium'] = float(pos['premium'])
                pos['status'] = pos.get('status', 'Active')
                pos['trade_date'] = pos.get('trade_date', datetime.now().strftime('%Y-%m-%d'))
                pos['id'] = len(self.positions)
                
                self.positions.append(pos)
            
            self.save_portfolio()
            return True
            
        except Exception as e:
            print(f"Error importing portfolio: {str(e)}")
            return False
    
    def clear_portfolio(self):
        """Clear all positions from portfolio"""
        try:
            self.positions = []
            self.trades_history = []
            self.save_portfolio()
            return True
        except Exception as e:
            print(f"Error clearing portfolio: {str(e)}")
            return False
    
    def get_performance_metrics(self):
        """
        Calculate portfolio performance metrics
        
        Returns:
            dict: Performance metrics
        """
        try:
            closed_positions = self.get_closed_positions()
            
            if not closed_positions:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'average_win': 0,
                    'average_loss': 0,
                    'profit_factor': 0,
                    'total_pnl': 0
                }
            
            winning_trades = 0
            losing_trades = 0
            total_wins = 0
            total_losses = 0
            total_pnl = 0
            
            for position in closed_positions:
                entry_premium = position['premium']
                exit_premium = position.get('closing_premium', entry_premium)
                
                if position['action'] == 'Buy':
                    pnl = (exit_premium - entry_premium) * position['quantity']
                else:
                    pnl = (entry_premium - exit_premium) * position['quantity']
                
                total_pnl += pnl
                
                if pnl > 0:
                    winning_trades += 1
                    total_wins += pnl
                elif pnl < 0:
                    losing_trades += 1
                    total_losses += abs(pnl)
            
            total_trades = len(closed_positions)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            avg_win = total_wins / winning_trades if winning_trades > 0 else 0
            avg_loss = total_losses / losing_trades if losing_trades > 0 else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'average_win': avg_win,
                'average_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_pnl': total_pnl
            }
            
        except Exception as e:
            print(f"Error calculating performance metrics: {str(e)}")
            return {}
