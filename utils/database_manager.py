"""
Database management utilities for Indian commodities options trading platform
"""
import os
import pandas as pd
from datetime import datetime, timedelta
import json
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import uuid

# Database configuration - SQLite local database
DATABASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'trading_data.db')
DATABASE_URL = f'sqlite:///{DATABASE_PATH}'

Base = declarative_base()

class DatabaseManager:
    """
    Manages database operations for the trading platform
    """
    
    def __init__(self):
        """Initialize database manager with SQLite"""
        try:
            # Create database directory if it doesn't exist
            db_dir = os.path.dirname(DATABASE_PATH)
            if not os.path.exists(db_dir):
                os.makedirs(db_dir)
                
            self.engine = create_engine(DATABASE_URL, echo=False)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self.metadata = MetaData()
            self.database_available = True
            print(f"SQLite database initialized at: {DATABASE_PATH}")
        except Exception as e:
            print(f"Database initialization failed: {e}")
            self.database_available = False
            self.engine = None
            self.SessionLocal = None
            self.metadata = None
        
    def initialize_database(self):
        """Create all tables if they don't exist"""
        if not self.database_available or not self.engine:
            print("Database not available. Skipping table creation.")
            return False
            
        try:
            with self.engine.connect() as conn:
                # Portfolio positions table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS portfolio_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT DEFAULT 'default_user',
                        commodity TEXT NOT NULL,
                        option_type TEXT NOT NULL,
                        action TEXT NOT NULL,
                        strike_price REAL NOT NULL,
                        quantity INTEGER NOT NULL,
                        premium REAL NOT NULL,
                        expiry_date TEXT NOT NULL,
                        trade_date TEXT NOT NULL,
                        status TEXT DEFAULT 'Active',
                        notes TEXT,
                        entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                        closing_premium REAL,
                        close_date TEXT,
                        close_time DATETIME
                    )
                """))
                
                # Trading recommendations history
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS recommendations_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        commodity TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        action TEXT NOT NULL,
                        strike_price TEXT,
                        expiry TEXT,
                        premium REAL,
                        max_loss REAL,
                        target_profit REAL,
                        risk_level TEXT,
                        confidence_score INTEGER,
                        market_bias TEXT,
                        reasoning TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        user_id TEXT DEFAULT 'default_user'
                    )
                """))
                
                # Market data cache
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS market_data_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date TEXT NOT NULL,
                        open_price REAL,
                        high_price REAL,
                        low_price REAL,
                        close_price REAL,
                        volume INTEGER,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, date)
                    )
                """))
                
                # User preferences
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(50) UNIQUE DEFAULT 'default_user',
                        risk_tolerance VARCHAR(20) DEFAULT 'moderate',
                        default_commodities TEXT,
                        notification_settings TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Risk metrics history
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS risk_metrics (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(50) DEFAULT 'default_user',
                        portfolio_var DECIMAL(15,2),
                        portfolio_delta DECIMAL(10,4),
                        portfolio_gamma DECIMAL(10,6),
                        portfolio_theta DECIMAL(10,2),
                        portfolio_vega DECIMAL(10,2),
                        total_margin DECIMAL(15,2),
                        portfolio_pnl DECIMAL(15,2),
                        calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                conn.commit()
                print("Database tables initialized successfully")
                return True
                
        except Exception as e:
            print(f"Error initializing database: {str(e)}")
            return False
    
    def add_portfolio_position(self, commodity, option_type, action, strike_price, 
                              quantity, premium, expiry_date, notes="", user_id="default_user"):
        """Add a new portfolio position to database"""
        if not self.database_available or not self.engine:
            print("Database not available. Position not saved to database.")
            return None
            
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    INSERT INTO portfolio_positions 
                    (user_id, commodity, option_type, action, strike_price, quantity, 
                     premium, expiry_date, trade_date, notes)
                    VALUES (:user_id, :commodity, :option_type, :action, :strike_price, 
                            :quantity, :premium, :expiry_date, :trade_date, :notes)
                    RETURNING id
                """), {
                    'user_id': user_id,
                    'commodity': commodity,
                    'option_type': option_type,
                    'action': action,
                    'strike_price': float(strike_price),
                    'quantity': int(quantity),
                    'premium': float(premium),
                    'expiry_date': expiry_date,
                    'trade_date': datetime.now().date(),
                    'notes': notes
                })
                
                position_id = result.fetchone()[0]
                conn.commit()
                return position_id
                
        except Exception as e:
            print(f"Error adding portfolio position: {str(e)}")
            return None
    
    def get_portfolio_positions(self, user_id="default_user", status="Active"):
        """Get portfolio positions from database"""
        if not self.database_available or not self.engine:
            return []
            
        try:
            with self.engine.connect() as conn:
                if status == "All":
                    result = conn.execute(text("""
                        SELECT * FROM portfolio_positions 
                        WHERE user_id = :user_id
                        ORDER BY entry_time DESC
                    """), {'user_id': user_id})
                else:
                    result = conn.execute(text("""
                        SELECT * FROM portfolio_positions 
                        WHERE user_id = :user_id AND status = :status
                        ORDER BY entry_time DESC
                    """), {'user_id': user_id, 'status': status})
                
                columns = result.keys()
                positions = []
                for row in result:
                    position = dict(zip(columns, row))
                    # Convert dates to strings for JSON serialization
                    if position['expiry_date']:
                        position['expiry_date'] = position['expiry_date'].strftime('%Y-%m-%d')
                    if position['trade_date']:
                        position['trade_date'] = position['trade_date'].strftime('%Y-%m-%d')
                    if position['close_date']:
                        position['close_date'] = position['close_date'].strftime('%Y-%m-%d')
                    positions.append(position)
                
                return positions
                
        except Exception as e:
            print(f"Error getting portfolio positions: {str(e)}")
            return []
    
    def close_portfolio_position(self, position_id, closing_premium=None, notes=""):
        """Close a portfolio position"""
        try:
            with self.engine.connect() as conn:
                update_data = {
                    'position_id': position_id,
                    'close_date': datetime.now().date(),
                    'close_time': datetime.now()
                }
                
                if closing_premium is not None:
                    update_data['closing_premium'] = float(closing_premium)
                
                if notes:
                    update_data['notes'] = notes
                
                query = """
                    UPDATE portfolio_positions 
                    SET status = 'Closed', close_date = :close_date, close_time = :close_time
                """
                
                if closing_premium is not None:
                    query += ", closing_premium = :closing_premium"
                
                if notes:
                    query += ", notes = COALESCE(notes, '') || ' | Close: ' || :notes"
                
                query += " WHERE id = :position_id"
                
                conn.execute(text(query), update_data)
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error closing position: {str(e)}")
            return False
    
    def save_recommendation(self, recommendation_data, user_id="default_user"):
        """Save a trading recommendation to database"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO recommendations_history 
                    (user_id, commodity, strategy_name, action, strike_price, expiry,
                     premium, max_loss, target_profit, risk_level, confidence_score,
                     market_bias, reasoning)
                    VALUES (:user_id, :commodity, :strategy_name, :action, :strike_price,
                            :expiry, :premium, :max_loss, :target_profit, :risk_level,
                            :confidence_score, :market_bias, :reasoning)
                """), {
                    'user_id': user_id,
                    'commodity': recommendation_data.get('commodity', ''),
                    'strategy_name': recommendation_data.get('strategy_name', ''),
                    'action': recommendation_data.get('action', ''),
                    'strike_price': str(recommendation_data.get('strike_price', '')),
                    'expiry': recommendation_data.get('expiry', ''),
                    'premium': float(recommendation_data.get('premium', 0)),
                    'max_loss': float(recommendation_data.get('max_loss', 0)) if isinstance(recommendation_data.get('max_loss'), (int, float)) else 0,
                    'target_profit': float(recommendation_data.get('target_profit', 0)) if isinstance(recommendation_data.get('target_profit'), (int, float)) else 0,
                    'risk_level': recommendation_data.get('risk_level', ''),
                    'confidence_score': int(recommendation_data.get('confidence_score', 0)),
                    'market_bias': recommendation_data.get('market_bias', ''),
                    'reasoning': recommendation_data.get('reasoning', '')
                })
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error saving recommendation: {str(e)}")
            return False
    
    def get_recommendations_history(self, user_id="default_user", days=30):
        """Get recommendations history from database"""
        try:
            with self.engine.connect() as conn:
                cutoff_date = datetime.now() - timedelta(days=days)
                
                result = conn.execute(text("""
                    SELECT * FROM recommendations_history 
                    WHERE user_id = :user_id AND created_at >= :cutoff_date
                    ORDER BY created_at DESC
                    LIMIT 100
                """), {'user_id': user_id, 'cutoff_date': cutoff_date})
                
                columns = result.keys()
                recommendations = []
                for row in result:
                    rec = dict(zip(columns, row))
                    if rec['created_at']:
                        rec['created_at'] = rec['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                    recommendations.append(rec)
                
                return recommendations
                
        except Exception as e:
            print(f"Error getting recommendations history: {str(e)}")
            return []
    
    def save_risk_metrics(self, risk_data, user_id="default_user"):
        """Save portfolio risk metrics to database"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO risk_metrics 
                    (user_id, portfolio_var, portfolio_delta, portfolio_gamma,
                     portfolio_theta, portfolio_vega, total_margin, portfolio_pnl)
                    VALUES (:user_id, :portfolio_var, :portfolio_delta, :portfolio_gamma,
                            :portfolio_theta, :portfolio_vega, :total_margin, :portfolio_pnl)
                """), {
                    'user_id': user_id,
                    'portfolio_var': float(risk_data.get('portfolio_var', 0)),
                    'portfolio_delta': float(risk_data.get('portfolio_delta', 0)),
                    'portfolio_gamma': float(risk_data.get('portfolio_gamma', 0)),
                    'portfolio_theta': float(risk_data.get('portfolio_theta', 0)),
                    'portfolio_vega': float(risk_data.get('portfolio_vega', 0)),
                    'total_margin': float(risk_data.get('total_margin', 0)),
                    'portfolio_pnl': float(risk_data.get('portfolio_pnl', 0))
                })
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error saving risk metrics: {str(e)}")
            return False
    
    def cache_market_data(self, symbol, data_df):
        """Cache market data in database"""
        try:
            with self.engine.connect() as conn:
                for index, row in data_df.iterrows():
                    conn.execute(text("""
                        INSERT INTO market_data_cache 
                        (symbol, date, open_price, high_price, low_price, close_price, volume)
                        VALUES (:symbol, :date, :open_price, :high_price, :low_price, :close_price, :volume)
                        ON CONFLICT (symbol, date) DO UPDATE SET
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        volume = EXCLUDED.volume,
                        updated_at = CURRENT_TIMESTAMP
                    """), {
                        'symbol': symbol,
                        'date': index.date(),
                        'open_price': float(row['Open']),
                        'high_price': float(row['High']),
                        'low_price': float(row['Low']),
                        'close_price': float(row['Close']),
                        'volume': int(row['Volume'])
                    })
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error caching market data: {str(e)}")
            return False
    
    def get_cached_market_data(self, symbol, days=30):
        """Get cached market data from database"""
        try:
            with self.engine.connect() as conn:
                cutoff_date = datetime.now().date() - timedelta(days=days)
                
                result = conn.execute(text("""
                    SELECT date, open_price, high_price, low_price, close_price, volume
                    FROM market_data_cache 
                    WHERE symbol = :symbol AND date >= :cutoff_date
                    ORDER BY date ASC
                """), {'symbol': symbol, 'cutoff_date': cutoff_date})
                
                data = []
                for row in result:
                    data.append({
                        'Date': row[0],
                        'Open': row[1],
                        'High': row[2],
                        'Low': row[3],
                        'Close': row[4],
                        'Volume': row[5]
                    })
                
                if data:
                    df = pd.DataFrame(data)
                    df.set_index('Date', inplace=True)
                    return df
                
                return None
                
        except Exception as e:
            print(f"Error getting cached market data: {str(e)}")
            return None
    
    def get_portfolio_summary(self, user_id="default_user"):
        """Get portfolio summary statistics"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_positions,
                        SUM(CASE WHEN status = 'Active' THEN 1 ELSE 0 END) as active_positions,
                        SUM(CASE WHEN status = 'Closed' THEN 1 ELSE 0 END) as closed_positions,
                        SUM(CASE WHEN status = 'Active' THEN premium * quantity ELSE 0 END) as active_investment,
                        AVG(CASE WHEN status = 'Closed' AND closing_premium IS NOT NULL 
                            THEN (closing_premium - premium) * quantity ELSE NULL END) as avg_closed_pnl
                    FROM portfolio_positions 
                    WHERE user_id = :user_id
                """), {'user_id': user_id})
                
                row = result.fetchone()
                if row:
                    return {
                        'total_positions': row[0] or 0,
                        'active_positions': row[1] or 0,
                        'closed_positions': row[2] or 0,
                        'active_investment': float(row[3] or 0),
                        'avg_closed_pnl': float(row[4] or 0)
                    }
                
                return {
                    'total_positions': 0,
                    'active_positions': 0,
                    'closed_positions': 0,
                    'active_investment': 0,
                    'avg_closed_pnl': 0
                }
                
        except Exception as e:
            print(f"Error getting portfolio summary: {str(e)}")
            return {}
    
    def cleanup_old_data(self, days=90):
        """Clean up old data from database"""
        try:
            with self.engine.connect() as conn:
                cutoff_date = datetime.now() - timedelta(days=days)
                
                # Clean old recommendations
                conn.execute(text("""
                    DELETE FROM recommendations_history 
                    WHERE created_at < :cutoff_date
                """), {'cutoff_date': cutoff_date})
                
                # Clean old risk metrics
                conn.execute(text("""
                    DELETE FROM risk_metrics 
                    WHERE calculated_at < :cutoff_date
                """), {'cutoff_date': cutoff_date})
                
                # Clean old market data cache
                conn.execute(text("""
                    DELETE FROM market_data_cache 
                    WHERE updated_at < :cutoff_date
                """), {'cutoff_date': cutoff_date})
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error cleaning up old data: {str(e)}")
            return False

# Global database manager instance
try:
    db_manager = DatabaseManager()
except Exception as e:
    print(f"Database initialization failed: {e}")
    # Create a minimal fallback manager
    class FallbackDatabaseManager:
        def __init__(self):
            self.database_available = False
        
        def get_portfolio_positions(self, user_id="default_user", status="Active"):
            return []
        
        def add_portfolio_position(self, *args, **kwargs):
            return None
        
        def save_recommendation(self, *args, **kwargs):
            return False
        
        def get_recommendations_history(self, *args, **kwargs):
            return []
        
        def save_risk_metrics(self, *args, **kwargs):
            return False
        
        def get_portfolio_summary(self, *args, **kwargs):
            return {'total_positions': 0, 'active_positions': 0, 'closed_positions': 0, 'active_investment': 0, 'avg_closed_pnl': 0}
        
        def close_portfolio_position(self, *args, **kwargs):
            return False
        
        def initialize_database(self):
            return False
    
    db_manager = FallbackDatabaseManager()