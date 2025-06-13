"""
Advanced AI-powered MCX Options Recommendation Engine
Uses free ML libraries: scikit-learn, ta (technical analysis), transformers
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import ta

# Technical Analysis
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
# Import available volume indicators
try:
    from ta.volume import VolumeSMAIndicator
except ImportError:
    VolumeSMAIndicator = None
try:
    from ta.volume import AccDistIndexIndicator  
except ImportError:
    AccDistIndexIndicator = None


class AdvancedMCXRecommendationEngine:
    """
    Advanced AI recommendation engine for MCX Crude Oil and Natural Gas options
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def calculate_advanced_technical_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        if df is None or len(df) < 50:
            return None
            
        # Price data
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df.get('Volume', pd.Series([1000000] * len(df)))
        
        # Trend Indicators
        df['SMA_5'] = SMAIndicator(close=close, window=5).sma_indicator()
        df['SMA_10'] = SMAIndicator(close=close, window=10).sma_indicator()
        df['SMA_20'] = SMAIndicator(close=close, window=20).sma_indicator()
        df['EMA_12'] = EMAIndicator(close=close, window=12).ema_indicator()
        df['EMA_26'] = EMAIndicator(close=close, window=26).ema_indicator()
        
        # MACD
        macd = MACD(close=close)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_histogram'] = macd.macd_diff()
        
        # Momentum Indicators
        df['RSI'] = RSIIndicator(close=close, window=14).rsi()
        df['RSI_oversold'] = (df['RSI'] < 30).astype(int)
        df['RSI_overbought'] = (df['RSI'] > 70).astype(int)
        
        # Stochastic
        stoch = StochasticOscillator(high=high, low=low, close=close)
        df['STOCH_K'] = stoch.stoch()
        df['STOCH_D'] = stoch.stoch_signal()
        
        # Volatility Indicators
        bb = BollingerBands(close=close, window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_squeeze'] = ((df['BB_upper'] - df['BB_lower']) / df['BB_middle'] < 0.1).astype(int)
        
        # ATR for volatility
        df['ATR'] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
        
        # Volume Indicators (with fallback)
        if VolumeSMAIndicator is not None:
            df['Volume_SMA'] = VolumeSMAIndicator(close=close, volume=volume, window=10).volume_sma()
        else:
            df['Volume_SMA'] = volume.rolling(10).mean()
            
        if AccDistIndexIndicator is not None:
            df['ADI'] = AccDistIndexIndicator(high=high, low=low, close=close, volume=volume).acc_dist_index()
        else:
            df['ADI'] = ((close - low) - (high - close)) / (high - low) * volume
        
        # Price patterns
        df['price_change'] = close.pct_change()
        df['price_momentum'] = close.rolling(5).mean() / close.rolling(20).mean() - 1
        df['volatility'] = close.rolling(10).std() / close.rolling(10).mean()
        
        # Support/Resistance levels
        df['resistance'] = high.rolling(20).max()
        df['support'] = low.rolling(20).min()
        df['near_resistance'] = (close >= df['resistance'] * 0.98).astype(int)
        df['near_support'] = (close <= df['support'] * 1.02).astype(int)
        
        return df
        
    def create_trading_features(self, df):
        """Create ML features for trading decisions"""
        features = []
        
        # Price-based features
        features.extend(['SMA_5', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26'])
        features.extend(['MACD', 'MACD_signal', 'MACD_histogram'])
        features.extend(['RSI', 'RSI_oversold', 'RSI_overbought'])
        features.extend(['STOCH_K', 'STOCH_D'])
        features.extend(['BB_squeeze', 'ATR'])
        features.extend(['price_momentum', 'volatility'])
        features.extend(['near_resistance', 'near_support'])
        
        return features
        
    def generate_market_signals(self, df):
        """Generate AI-powered market signals"""
        if df is None or len(df) < 50:
            return {
                'signal': 'HOLD',
                'confidence': 0.3,
                'strength': 'LOW',
                'entry_zone': None,
                'targets': [],
                'stop_loss': None
            }
            
        # Calculate features
        df = self.calculate_advanced_technical_indicators(df)
        features = self.create_trading_features(df)
        
        # Get latest values
        latest = df.iloc[-1]
        current_price = latest['Close']
        
        # Signal calculation using multiple indicators
        signals = []
        
        # Trend signals
        if latest['SMA_5'] > latest['SMA_20'] and latest['MACD'] > latest['MACD_signal']:
            signals.append('BUY')
        elif latest['SMA_5'] < latest['SMA_20'] and latest['MACD'] < latest['MACD_signal']:
            signals.append('SELL')
            
        # Momentum signals
        if latest['RSI'] < 30 and latest['STOCH_K'] < 20:
            signals.append('BUY')
        elif latest['RSI'] > 70 and latest['STOCH_K'] > 80:
            signals.append('SELL')
            
        # Volatility breakout
        if latest['BB_squeeze'] == 1 and latest['price_momentum'] > 0.02:
            signals.append('BUY')
            
        # Support/Resistance
        if latest['near_support'] == 1 and latest['RSI'] < 40:
            signals.append('BUY')
        elif latest['near_resistance'] == 1 and latest['RSI'] > 60:
            signals.append('SELL')
            
        # Determine overall signal
        buy_signals = signals.count('BUY')
        sell_signals = signals.count('SELL')
        total_signals = len(signals)
        
        if buy_signals > sell_signals:
            signal = 'BUY'
            confidence = buy_signals / max(total_signals, 1)
        elif sell_signals > buy_signals:
            signal = 'SELL'
            confidence = sell_signals / max(total_signals, 1)
        else:
            signal = 'HOLD'
            confidence = 0.3
            
        # Calculate strength
        if confidence >= 0.7:
            strength = 'HIGH'
        elif confidence >= 0.5:
            strength = 'MEDIUM'
        else:
            strength = 'LOW'
            
        # Calculate entry zones and targets
        atr = latest['ATR']
        
        if signal == 'BUY':
            entry_zone = [current_price * 0.998, current_price * 1.002]
            targets = [current_price * 1.01, current_price * 1.02, current_price * 1.03]
            stop_loss = current_price - (atr * 1.5)
        elif signal == 'SELL':
            entry_zone = [current_price * 0.998, current_price * 1.002]
            targets = [current_price * 0.99, current_price * 0.98, current_price * 0.97]
            stop_loss = current_price + (atr * 1.5)
        else:
            entry_zone = None
            targets = []
            stop_loss = None
            
        return {
            'signal': signal,
            'confidence': confidence,
            'strength': strength,
            'entry_zone': entry_zone,
            'targets': targets,
            'stop_loss': stop_loss,
            'rsi': latest['RSI'],
            'macd': latest['MACD'],
            'bb_position': 'UPPER' if current_price > latest['BB_upper'] else 'LOWER' if current_price < latest['BB_lower'] else 'MIDDLE'
        }
        
    def generate_options_recommendations(self, commodity, current_price, market_signals):
        """Generate specific options trading recommendations"""
        recommendations = []
        
        signal = market_signals['signal']
        confidence = market_signals['confidence']
        strength = market_signals['strength']
        
        if commodity == 'Crude Oil':
            lot_size = 100
            if signal == 'BUY':
                # Call options
                strike_atm = int(current_price / 100) * 100
                strike_otm = strike_atm + 100
                
                recommendations.append({
                    'type': 'CALL',
                    'strike': strike_atm,
                    'instrument': f'CRUDEOIL {strike_atm} CE',
                    'strategy': 'ATM Call for bullish momentum',
                    'confidence': strength,
                    'risk_level': 'MEDIUM',
                    'entry_price_range': market_signals['entry_zone'],
                    'targets': market_signals['targets'],
                    'stop_loss': market_signals['stop_loss'],
                    'max_loss': f"Premium paid (approx ₹50-100 per lot)",
                    'reasoning': f"RSI: {market_signals['rsi']:.1f}, MACD bullish crossover, BB position: {market_signals['bb_position']}"
                })
                
                if confidence > 0.6:
                    recommendations.append({
                        'type': 'CALL',
                        'strike': strike_otm,
                        'instrument': f'CRUDEOIL {strike_otm} CE',
                        'strategy': 'OTM Call for high conviction',
                        'confidence': 'HIGH',
                        'risk_level': 'HIGH',
                        'entry_price_range': market_signals['entry_zone'],
                        'targets': [current_price + 200, current_price + 300],
                        'stop_loss': market_signals['stop_loss'],
                        'max_loss': f"Premium paid (approx ₹20-40 per lot)",
                        'reasoning': f"Strong bullish signals, high confidence ({confidence:.1%})"
                    })
                    
            elif signal == 'SELL':
                # Put options
                strike_atm = int(current_price / 100) * 100
                strike_otm = strike_atm - 100
                
                recommendations.append({
                    'type': 'PUT',
                    'strike': strike_atm,
                    'instrument': f'CRUDEOIL {strike_atm} PE',
                    'strategy': 'ATM Put for bearish momentum',
                    'confidence': strength,
                    'risk_level': 'MEDIUM',
                    'entry_price_range': market_signals['entry_zone'],
                    'targets': market_signals['targets'],
                    'stop_loss': market_signals['stop_loss'],
                    'max_loss': f"Premium paid (approx ₹50-100 per lot)",
                    'reasoning': f"RSI: {market_signals['rsi']:.1f}, MACD bearish crossover, BB position: {market_signals['bb_position']}"
                })
                
        elif commodity == 'Natural Gas':
            lot_size = 1250
            if signal == 'BUY':
                strike_atm = int(current_price / 5) * 5
                strike_otm = strike_atm + 5
                
                recommendations.append({
                    'type': 'CALL',
                    'strike': strike_atm,
                    'instrument': f'NATURALGAS {strike_atm} CE',
                    'strategy': 'ATM Call for gas rally',
                    'confidence': strength,
                    'risk_level': 'MEDIUM',
                    'entry_price_range': market_signals['entry_zone'],
                    'targets': market_signals['targets'],
                    'stop_loss': market_signals['stop_loss'],
                    'max_loss': f"Premium paid (approx ₹5-15 per lot)",
                    'reasoning': f"RSI: {market_signals['rsi']:.1f}, Technical breakout signals"
                })
                
            elif signal == 'SELL':
                strike_atm = int(current_price / 5) * 5
                strike_otm = strike_atm - 5
                
                recommendations.append({
                    'type': 'PUT',
                    'strike': strike_atm,
                    'instrument': f'NATURALGAS {strike_atm} PE',
                    'strategy': 'ATM Put for gas decline',
                    'confidence': strength,
                    'risk_level': 'MEDIUM',
                    'entry_price_range': market_signals['entry_zone'],
                    'targets': market_signals['targets'],
                    'stop_loss': market_signals['stop_loss'],
                    'max_loss': f"Premium paid (approx ₹5-15 per lot)",
                    'reasoning': f"RSI: {market_signals['rsi']:.1f}, Bearish momentum confirmed"
                })
                
        return recommendations
        
    def get_risk_management_rules(self, commodity, recommendations):
        """Generate risk management guidelines"""
        rules = {
            'position_sizing': f"Risk only 2-3% of capital per trade",
            'stop_loss': "Exit if underlying moves against you by 1.5x ATR",
            'profit_booking': "Book 50% profits at first target, trail rest",
            'time_decay': "Exit options 1 week before expiry if not profitable",
            'max_positions': f"Hold maximum 2-3 {commodity} positions simultaneously"
        }
        
        return rules


# Initialize the AI engine
ai_engine = AdvancedMCXRecommendationEngine()