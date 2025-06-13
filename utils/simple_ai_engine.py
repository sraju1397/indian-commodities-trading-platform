"""
Simplified AI-powered MCX Options Recommendation Engine
Uses core technical analysis with scikit-learn for signal generation
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core ML and Technical Analysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange


class SimplifiedMCXRecommendationEngine:
    """
    Simplified AI recommendation engine for MCX options
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def calculate_technical_indicators(self, df):
        """Calculate essential technical indicators"""
        if df is None or len(df) < 30:
            return None
            
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Moving Averages
        df['SMA_10'] = SMAIndicator(close=close, window=10).sma_indicator()
        df['SMA_20'] = SMAIndicator(close=close, window=20).sma_indicator()
        df['EMA_12'] = EMAIndicator(close=close, window=12).ema_indicator()
        df['EMA_26'] = EMAIndicator(close=close, window=26).ema_indicator()
        
        # MACD
        macd = MACD(close=close)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_histogram'] = macd.macd_diff()
        
        # RSI
        df['RSI'] = RSIIndicator(close=close, window=14).rsi()
        
        # Stochastic
        stoch = StochasticOscillator(high=high, low=low, close=close)
        df['STOCH_K'] = stoch.stoch()
        df['STOCH_D'] = stoch.stoch_signal()
        
        # Bollinger Bands
        bb = BollingerBands(close=close, window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_middle'] = bb.bollinger_mavg()
        
        # ATR
        df['ATR'] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
        
        # Price patterns
        df['price_momentum'] = close.pct_change(5)
        df['volatility'] = close.rolling(10).std() / close.rolling(10).mean()
        
        return df
        
    def generate_trading_signals(self, df):
        """Generate trading signals using technical analysis"""
        if df is None or len(df) < 30:
            return {
                'signal': 'HOLD',
                'confidence': 'LOW',
                'strength': 0.3,
                'entry_zone': None,
                'targets': [],
                'stop_loss': None,
                'reasoning': 'Insufficient data for analysis'
            }
            
        # Calculate indicators
        df = self.calculate_technical_indicators(df)
        if df is None:
            return self._default_signal()
            
        # Get latest values
        latest = df.iloc[-1]
        current_price = latest['Close']
        
        # Signal scoring system
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        reasons = []
        
        # Trend Analysis
        if latest['SMA_10'] > latest['SMA_20']:
            bullish_signals += 2
            reasons.append("Short-term trend bullish")
        else:
            bearish_signals += 2
            reasons.append("Short-term trend bearish")
        total_signals += 2
        
        # MACD Analysis
        if latest['MACD'] > latest['MACD_signal'] and latest['MACD_histogram'] > 0:
            bullish_signals += 2
            reasons.append("MACD bullish crossover")
        elif latest['MACD'] < latest['MACD_signal'] and latest['MACD_histogram'] < 0:
            bearish_signals += 2
            reasons.append("MACD bearish crossover")
        total_signals += 2
        
        # RSI Analysis
        if latest['RSI'] < 30:
            bullish_signals += 3
            reasons.append(f"RSI oversold at {latest['RSI']:.1f}")
        elif latest['RSI'] > 70:
            bearish_signals += 3
            reasons.append(f"RSI overbought at {latest['RSI']:.1f}")
        elif 30 <= latest['RSI'] <= 40:
            bullish_signals += 1
            reasons.append("RSI showing buying opportunity")
        elif 60 <= latest['RSI'] <= 70:
            bearish_signals += 1
            reasons.append("RSI showing selling opportunity")
        total_signals += 3
        
        # Stochastic Analysis
        if latest['STOCH_K'] < 20 and latest['STOCH_D'] < 20:
            bullish_signals += 1
            reasons.append("Stochastic oversold")
        elif latest['STOCH_K'] > 80 and latest['STOCH_D'] > 80:
            bearish_signals += 1
            reasons.append("Stochastic overbought")
        total_signals += 1
        
        # Bollinger Bands Analysis
        if current_price <= latest['BB_lower']:
            bullish_signals += 2
            reasons.append("Price at lower Bollinger Band")
        elif current_price >= latest['BB_upper']:
            bearish_signals += 2
            reasons.append("Price at upper Bollinger Band")
        total_signals += 2
        
        # Determine signal
        if bullish_signals > bearish_signals:
            signal = 'BUY'
            strength = bullish_signals / total_signals
        elif bearish_signals > bullish_signals:
            signal = 'SELL'
            strength = bearish_signals / total_signals
        else:
            signal = 'HOLD'
            strength = 0.3
            
        # Confidence levels
        if strength >= 0.7:
            confidence = 'HIGH'
        elif strength >= 0.5:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
            
        # Calculate targets and stop loss
        atr = latest['ATR']
        
        if signal == 'BUY':
            entry_zone = [current_price * 0.999, current_price * 1.001]
            targets = [
                current_price + (atr * 1.5),
                current_price + (atr * 2.5),
                current_price + (atr * 3.5)
            ]
            stop_loss = current_price - (atr * 2)
        elif signal == 'SELL':
            entry_zone = [current_price * 0.999, current_price * 1.001]
            targets = [
                current_price - (atr * 1.5),
                current_price - (atr * 2.5),
                current_price - (atr * 3.5)
            ]
            stop_loss = current_price + (atr * 2)
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
            'reasoning': '; '.join(reasons[:3]),  # Top 3 reasons
            'rsi': latest['RSI'],
            'macd': latest['MACD'],
            'atr': atr,
            'bb_position': self._get_bb_position(current_price, latest)
        }
        
    def _get_bb_position(self, price, latest):
        """Determine position relative to Bollinger Bands"""
        if price >= latest['BB_upper']:
            return 'UPPER'
        elif price <= latest['BB_lower']:
            return 'LOWER'
        else:
            return 'MIDDLE'
            
    def _default_signal(self):
        """Default signal when analysis fails"""
        return {
            'signal': 'HOLD',
            'confidence': 'LOW',
            'strength': 0.3,
            'entry_zone': None,
            'targets': [],
            'stop_loss': None,
            'reasoning': 'Analysis unavailable',
            'rsi': 50,
            'macd': 0,
            'atr': 0,
            'bb_position': 'MIDDLE'
        }
        
    def generate_options_recommendations(self, commodity, current_price, signals):
        """Generate specific options recommendations"""
        recommendations = []
        
        signal = signals['signal']
        confidence = signals['confidence']
        
        if commodity == 'Crude Oil':
            if signal == 'BUY':
                # ATM Call
                strike_atm = int(current_price / 100) * 100
                recommendations.append({
                    'type': 'CALL',
                    'strike': strike_atm,
                    'instrument': f'CRUDEOIL {strike_atm} CE',
                    'strategy': 'ATM Call for bullish momentum',
                    'confidence': confidence,
                    'risk_level': 'MEDIUM',
                    'entry_zone': signals['entry_zone'],
                    'targets': signals['targets'],
                    'stop_loss': signals['stop_loss'],
                    'max_loss': 'Premium paid (₹50-100 per lot)',
                    'reasoning': signals['reasoning']
                })
                
                # OTM Call for high confidence
                if confidence == 'HIGH':
                    strike_otm = strike_atm + 100
                    recommendations.append({
                        'type': 'CALL',
                        'strike': strike_otm,
                        'instrument': f'CRUDEOIL {strike_otm} CE',
                        'strategy': 'OTM Call for high conviction',
                        'confidence': 'HIGH',
                        'risk_level': 'HIGH',
                        'entry_zone': signals['entry_zone'],
                        'targets': [t + 100 for t in signals['targets']],
                        'stop_loss': signals['stop_loss'],
                        'max_loss': 'Premium paid (₹20-40 per lot)',
                        'reasoning': f"High confidence signal: {signals['reasoning']}"
                    })
                    
            elif signal == 'SELL':
                # ATM Put
                strike_atm = int(current_price / 100) * 100
                recommendations.append({
                    'type': 'PUT',
                    'strike': strike_atm,
                    'instrument': f'CRUDEOIL {strike_atm} PE',
                    'strategy': 'ATM Put for bearish momentum',
                    'confidence': confidence,
                    'risk_level': 'MEDIUM',
                    'entry_zone': signals['entry_zone'],
                    'targets': signals['targets'],
                    'stop_loss': signals['stop_loss'],
                    'max_loss': 'Premium paid (₹50-100 per lot)',
                    'reasoning': signals['reasoning']
                })
                
        elif commodity == 'Natural Gas':
            if signal == 'BUY':
                strike_atm = int(current_price / 5) * 5
                recommendations.append({
                    'type': 'CALL',
                    'strike': strike_atm,
                    'instrument': f'NATURALGAS {strike_atm} CE',
                    'strategy': 'ATM Call for gas rally',
                    'confidence': confidence,
                    'risk_level': 'MEDIUM',
                    'entry_zone': signals['entry_zone'],
                    'targets': signals['targets'],
                    'stop_loss': signals['stop_loss'],
                    'max_loss': 'Premium paid (₹5-15 per lot)',
                    'reasoning': signals['reasoning']
                })
                
            elif signal == 'SELL':
                strike_atm = int(current_price / 5) * 5
                recommendations.append({
                    'type': 'PUT',
                    'strike': strike_atm,
                    'instrument': f'NATURALGAS {strike_atm} PE',
                    'strategy': 'ATM Put for gas decline',
                    'confidence': confidence,
                    'risk_level': 'MEDIUM',
                    'entry_zone': signals['entry_zone'],
                    'targets': signals['targets'],
                    'stop_loss': signals['stop_loss'],
                    'max_loss': 'Premium paid (₹5-15 per lot)',
                    'reasoning': signals['reasoning']
                })
                
        return recommendations


# Initialize the simplified AI engine
simple_ai_engine = SimplifiedMCXRecommendationEngine()