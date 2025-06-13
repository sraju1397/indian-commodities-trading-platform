"""
Advanced options recommendation engine with risk and confidence analysis
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from utils.options_calculator import black_scholes_price, calculate_greeks
from utils.data_fetcher import get_commodity_data, get_historical_volatility

class OptionsRecommendationEngine:
    """
    Generate specific options trading recommendations with risk and confidence levels
    """
    
    def __init__(self):
        self.risk_free_rate = 0.065  # 6.5% RBI rate
        
    def analyze_commodity_and_recommend(self, commodity, symbol, risk_tolerance='moderate'):
        """
        Analyze a commodity and provide specific options recommendations
        
        Args:
            commodity (str): Commodity name
            symbol (str): Commodity symbol
            risk_tolerance (str): 'conservative', 'moderate', 'aggressive'
        
        Returns:
            dict: Comprehensive recommendations with specific strikes and strategies
        """
        try:
            # Get market data
            data = get_commodity_data(symbol, period='3mo')
            if data is None or data.empty:
                return self._generate_fallback_recommendation(commodity, risk_tolerance)
            
            current_price = data['Close'].iloc[-1]
            
            # Calculate technical indicators
            technical_analysis = self._calculate_technical_indicators(data)
            
            # Get volatility
            historical_vol = get_historical_volatility(symbol, period='1mo') / 100
            implied_vol = max(historical_vol, 0.15)  # Minimum 15% IV
            
            # Generate expiry dates
            expiry_dates = self._get_optimal_expiry_dates()
            
            # Create recommendations based on analysis
            recommendations = []
            
            # Determine market bias
            market_bias = self._determine_market_bias(technical_analysis)
            
            for expiry_name, expiry_date in expiry_dates.items():
                time_to_expiry = (expiry_date - datetime.now()).days / 365.0
                
                if time_to_expiry <= 0:
                    continue
                
                # Generate strategy recommendations
                strategy_recs = self._generate_strategy_recommendations(
                    commodity, current_price, time_to_expiry, implied_vol,
                    market_bias, technical_analysis, risk_tolerance, expiry_name
                )
                
                recommendations.extend(strategy_recs)
            
            # Sort by confidence and filter top recommendations
            recommendations = sorted(recommendations, key=lambda x: x['confidence_score'], reverse=True)
            
            return {
                'commodity': commodity,
                'current_price': current_price,
                'market_bias': market_bias,
                'technical_summary': technical_analysis,
                'implied_volatility': implied_vol * 100,
                'recommendations': recommendations[:6],  # Top 6 recommendations
                'market_conditions': self._assess_market_conditions(technical_analysis, implied_vol)
            }
            
        except Exception as e:
            print(f"Error in recommendation engine: {str(e)}")
            return self._generate_fallback_recommendation(commodity, risk_tolerance)
    
    def _calculate_technical_indicators(self, data):
        """Calculate comprehensive technical indicators"""
        try:
            # Price-based indicators
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
            price_change_pct = ((current_price - prev_price) / prev_price) * 100
            
            # Moving averages
            sma_20 = data['Close'].rolling(20).mean().iloc[-1] if len(data) >= 20 else current_price
            sma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else current_price
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1] if len(gain) >= 14 else 50
            
            # Volatility
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 10 else 25
            
            # Support and resistance
            high_20 = data['High'].rolling(20).max().iloc[-1] if len(data) >= 20 else current_price
            low_20 = data['Low'].rolling(20).min().iloc[-1] if len(data) >= 20 else current_price
            
            return {
                'current_price': current_price,
                'price_change_pct': price_change_pct,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi,
                'volatility': volatility,
                'resistance': high_20,
                'support': low_20,
                'trend_strength': abs(price_change_pct),
                'volume_trend': data['Volume'].rolling(5).mean().iloc[-1] / data['Volume'].rolling(20).mean().iloc[-1] if len(data) >= 20 else 1.0
            }
            
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            return {
                'current_price': 50000,
                'price_change_pct': 0,
                'sma_20': 50000,
                'sma_50': 50000,
                'rsi': 50,
                'volatility': 25,
                'resistance': 52000,
                'support': 48000,
                'trend_strength': 1,
                'volume_trend': 1.0
            }
    
    def _determine_market_bias(self, technical_analysis):
        """Determine overall market bias"""
        signals = []
        
        # Price vs moving averages
        if technical_analysis['current_price'] > technical_analysis['sma_20']:
            signals.append(1)
        else:
            signals.append(-1)
            
        if technical_analysis['sma_20'] > technical_analysis['sma_50']:
            signals.append(1)
        else:
            signals.append(-1)
        
        # RSI analysis
        if technical_analysis['rsi'] < 30:
            signals.append(1)  # Oversold = bullish
        elif technical_analysis['rsi'] > 70:
            signals.append(-1)  # Overbought = bearish
        else:
            signals.append(0)  # Neutral
        
        # Price momentum
        if technical_analysis['price_change_pct'] > 2:
            signals.append(1)
        elif technical_analysis['price_change_pct'] < -2:
            signals.append(-1)
        else:
            signals.append(0)
        
        avg_signal = np.mean(signals)
        
        if avg_signal > 0.3:
            return 'bullish'
        elif avg_signal < -0.3:
            return 'bearish'
        else:
            return 'neutral'
    
    def _get_optimal_expiry_dates(self):
        """Get optimal expiry dates for recommendations"""
        today = datetime.now()
        
        # Weekly expiry (next Thursday)
        days_to_thursday = (3 - today.weekday()) % 7
        if days_to_thursday == 0:
            days_to_thursday = 7
        weekly_expiry = today + timedelta(days=days_to_thursday)
        
        # Monthly expiry (4 weeks out)
        monthly_expiry = today + timedelta(days=28)
        
        return {
            'Weekly': weekly_expiry,
            'Monthly': monthly_expiry
        }
    
    def _generate_strategy_recommendations(self, commodity, current_price, time_to_expiry, 
                                         implied_vol, market_bias, technical_analysis, 
                                         risk_tolerance, expiry_name):
        """Generate specific strategy recommendations"""
        recommendations = []
        
        # ATM strike (rounded to nearest 50)
        atm_strike = round(current_price / 50) * 50
        
        if market_bias == 'bullish':
            recommendations.extend(self._generate_bullish_strategies(
                commodity, current_price, atm_strike, time_to_expiry, 
                implied_vol, technical_analysis, risk_tolerance, expiry_name
            ))
        
        elif market_bias == 'bearish':
            recommendations.extend(self._generate_bearish_strategies(
                commodity, current_price, atm_strike, time_to_expiry,
                implied_vol, technical_analysis, risk_tolerance, expiry_name
            ))
        
        else:  # neutral
            recommendations.extend(self._generate_neutral_strategies(
                commodity, current_price, atm_strike, time_to_expiry,
                implied_vol, technical_analysis, risk_tolerance, expiry_name
            ))
        
        return recommendations
    
    def _generate_bullish_strategies(self, commodity, current_price, atm_strike, 
                                   time_to_expiry, implied_vol, technical_analysis, 
                                   risk_tolerance, expiry_name):
        """Generate bullish strategy recommendations"""
        strategies = []
        
        # Strategy 1: ATM Call Buy (Aggressive)
        if risk_tolerance in ['moderate', 'aggressive']:
            call_price = black_scholes_price(current_price, atm_strike, time_to_expiry, 
                                           self.risk_free_rate, implied_vol, 'call')
            call_greeks = calculate_greeks(current_price, atm_strike, time_to_expiry,
                                         self.risk_free_rate, implied_vol, 'call')
            
            target_price = current_price * 1.08
            confidence = self._calculate_confidence(technical_analysis, 'bullish', time_to_expiry)
            risk_level = 'High' if risk_tolerance == 'aggressive' else 'Medium'
            
            strategies.append({
                'strategy_name': f'Long {atm_strike} Call',
                'action': 'BUY',
                'option_type': 'Call',
                'strike_price': atm_strike,
                'expiry': expiry_name,
                'premium': call_price,
                'quantity_suggested': 1,
                'max_loss': call_price,
                'target_profit': target_price - atm_strike - call_price,
                'breakeven': atm_strike + call_price,
                'target_price': target_price,
                'stop_loss': current_price * 0.97,
                'risk_level': risk_level,
                'confidence_score': confidence,
                'probability_of_profit': self._calculate_pop(current_price, atm_strike, implied_vol, time_to_expiry, 'call'),
                'delta': call_greeks['delta'],
                'gamma': call_greeks['gamma'],
                'theta': call_greeks['theta'],
                'reasoning': f"Strong bullish signals with RSI at {technical_analysis['rsi']:.1f}. Price above SMA20 ({technical_analysis['sma_20']:.0f}). Target: 8% upside."
            })
        
        # Strategy 2: Bull Call Spread (Conservative to Moderate)
        if risk_tolerance in ['conservative', 'moderate']:
            otm_strike = atm_strike + 100
            
            long_call_price = black_scholes_price(current_price, atm_strike, time_to_expiry,
                                                self.risk_free_rate, implied_vol, 'call')
            short_call_price = black_scholes_price(current_price, otm_strike, time_to_expiry,
                                                 self.risk_free_rate, implied_vol, 'call')
            
            net_debit = long_call_price - short_call_price
            max_profit = (otm_strike - atm_strike) - net_debit
            
            confidence = self._calculate_confidence(technical_analysis, 'bullish', time_to_expiry) * 0.9
            risk_level = 'Low' if risk_tolerance == 'conservative' else 'Medium'
            
            strategies.append({
                'strategy_name': f'Bull Call Spread {atm_strike}/{otm_strike}',
                'action': f'BUY {atm_strike} Call, SELL {otm_strike} Call',
                'option_type': 'Spread',
                'strike_price': f'{atm_strike}/{otm_strike}',
                'expiry': expiry_name,
                'premium': net_debit,
                'quantity_suggested': 1,
                'max_loss': net_debit,
                'target_profit': max_profit,
                'breakeven': atm_strike + net_debit,
                'target_price': otm_strike,
                'stop_loss': current_price * 0.98,
                'risk_level': risk_level,
                'confidence_score': confidence,
                'probability_of_profit': self._calculate_spread_pop(current_price, atm_strike, otm_strike, implied_vol, time_to_expiry),
                'reasoning': f"Limited risk bullish strategy. Profit if {commodity} moves to {otm_strike} by expiry. Conservative approach to bullish bias."
            })
        
        return strategies
    
    def _generate_bearish_strategies(self, commodity, current_price, atm_strike,
                                   time_to_expiry, implied_vol, technical_analysis,
                                   risk_tolerance, expiry_name):
        """Generate bearish strategy recommendations"""
        strategies = []
        
        # Strategy 1: ATM Put Buy
        if risk_tolerance in ['moderate', 'aggressive']:
            put_price = black_scholes_price(current_price, atm_strike, time_to_expiry,
                                          self.risk_free_rate, implied_vol, 'put')
            put_greeks = calculate_greeks(current_price, atm_strike, time_to_expiry,
                                        self.risk_free_rate, implied_vol, 'put')
            
            target_price = current_price * 0.92
            confidence = self._calculate_confidence(technical_analysis, 'bearish', time_to_expiry)
            risk_level = 'High' if risk_tolerance == 'aggressive' else 'Medium'
            
            strategies.append({
                'strategy_name': f'Long {atm_strike} Put',
                'action': 'BUY',
                'option_type': 'Put',
                'strike_price': atm_strike,
                'expiry': expiry_name,
                'premium': put_price,
                'quantity_suggested': 1,
                'max_loss': put_price,
                'target_profit': atm_strike - target_price - put_price,
                'breakeven': atm_strike - put_price,
                'target_price': target_price,
                'stop_loss': current_price * 1.03,
                'risk_level': risk_level,
                'confidence_score': confidence,
                'probability_of_profit': self._calculate_pop(current_price, atm_strike, implied_vol, time_to_expiry, 'put'),
                'delta': put_greeks['delta'],
                'gamma': put_greeks['gamma'],
                'theta': put_greeks['theta'],
                'reasoning': f"Bearish signals detected. RSI at {technical_analysis['rsi']:.1f}, price below key levels. Target: 8% downside."
            })
        
        # Strategy 2: Bear Put Spread
        if risk_tolerance in ['conservative', 'moderate']:
            otm_strike = atm_strike - 100
            
            long_put_price = black_scholes_price(current_price, atm_strike, time_to_expiry,
                                               self.risk_free_rate, implied_vol, 'put')
            short_put_price = black_scholes_price(current_price, otm_strike, time_to_expiry,
                                                self.risk_free_rate, implied_vol, 'put')
            
            net_debit = long_put_price - short_put_price
            max_profit = (atm_strike - otm_strike) - net_debit
            
            confidence = self._calculate_confidence(technical_analysis, 'bearish', time_to_expiry) * 0.9
            
            strategies.append({
                'strategy_name': f'Bear Put Spread {atm_strike}/{otm_strike}',
                'action': f'BUY {atm_strike} Put, SELL {otm_strike} Put',
                'option_type': 'Spread',
                'strike_price': f'{atm_strike}/{otm_strike}',
                'expiry': expiry_name,
                'premium': net_debit,
                'quantity_suggested': 1,
                'max_loss': net_debit,
                'target_profit': max_profit,
                'breakeven': atm_strike - net_debit,
                'target_price': otm_strike,
                'stop_loss': current_price * 1.02,
                'risk_level': 'Medium',
                'confidence_score': confidence,
                'probability_of_profit': self._calculate_spread_pop(current_price, otm_strike, atm_strike, implied_vol, time_to_expiry),
                'reasoning': f"Limited risk bearish strategy. Profit if {commodity} declines to {otm_strike}. Controlled downside approach."
            })
        
        return strategies
    
    def _generate_neutral_strategies(self, commodity, current_price, atm_strike,
                                   time_to_expiry, implied_vol, technical_analysis,
                                   risk_tolerance, expiry_name):
        """Generate neutral/sideways strategy recommendations"""
        strategies = []
        
        # Strategy 1: Iron Condor (for neutral markets)
        if time_to_expiry > 0.05:  # At least ~2 weeks
            put_strike_short = atm_strike - 50
            put_strike_long = atm_strike - 100
            call_strike_short = atm_strike + 50
            call_strike_long = atm_strike + 100
            
            # Calculate net credit (simplified)
            net_credit = 25  # Estimated based on typical spreads
            max_profit = net_credit
            max_loss = 50 - net_credit  # Width of wings minus credit
            
            confidence = self._calculate_confidence(technical_analysis, 'neutral', time_to_expiry)
            
            strategies.append({
                'strategy_name': f'Iron Condor {put_strike_short}/{call_strike_short}',
                'action': f'SELL {put_strike_short}P, BUY {put_strike_long}P, SELL {call_strike_short}C, BUY {call_strike_long}C',
                'option_type': 'Iron Condor',
                'strike_price': f'{put_strike_short}/{put_strike_long}/{call_strike_short}/{call_strike_long}',
                'expiry': expiry_name,
                'premium': net_credit,
                'quantity_suggested': 1,
                'max_loss': max_loss,
                'target_profit': max_profit,
                'breakeven': f'{put_strike_short - net_credit} to {call_strike_short + net_credit}',
                'target_price': f'{put_strike_short} to {call_strike_short}',
                'stop_loss': f'Exit if price moves outside {put_strike_short - 25} to {call_strike_short + 25}',
                'risk_level': 'Medium',
                'confidence_score': confidence,
                'probability_of_profit': 65,  # Typical for Iron Condors
                'reasoning': f"Neutral market conditions. Profit from time decay if {commodity} stays between {put_strike_short} and {call_strike_short}."
            })
        
        # Strategy 2: Short Straddle (for very low volatility)
        if implied_vol < 0.20 and risk_tolerance == 'aggressive':
            straddle_premium = black_scholes_price(current_price, atm_strike, time_to_expiry,
                                                 self.risk_free_rate, implied_vol, 'call') + \
                             black_scholes_price(current_price, atm_strike, time_to_expiry,
                                                self.risk_free_rate, implied_vol, 'put')
            
            confidence = self._calculate_confidence(technical_analysis, 'neutral', time_to_expiry)
            
            strategies.append({
                'strategy_name': f'Short Straddle {atm_strike}',
                'action': f'SELL {atm_strike} Call and SELL {atm_strike} Put',
                'option_type': 'Short Straddle',
                'strike_price': atm_strike,
                'expiry': expiry_name,
                'premium': straddle_premium,
                'quantity_suggested': 1,
                'max_loss': 'Unlimited',
                'target_profit': straddle_premium,
                'breakeven': f'{atm_strike - straddle_premium} to {atm_strike + straddle_premium}',
                'target_price': atm_strike,
                'stop_loss': f'Exit if move beyond ±{straddle_premium * 0.6:.0f}',
                'risk_level': 'Very High',
                'confidence_score': confidence * 0.8,
                'probability_of_profit': 40,
                'reasoning': f"Low volatility environment. Profit from time decay if {commodity} stays near {atm_strike}. High risk strategy."
            })
        
        return strategies
    
    def _calculate_confidence(self, technical_analysis, bias, time_to_expiry):
        """Enhanced confidence calculation with multiple factors"""
        confidence = 45  # Slightly lower base for more realistic scoring
        
        # RSI factor with refined thresholds
        rsi = technical_analysis.get('rsi', 50)
        if bias == 'bullish':
            if rsi < 30:  # Oversold
                confidence += 20
            elif rsi < 40:
                confidence += 15
            elif rsi < 50:
                confidence += 8
        elif bias == 'bearish':
            if rsi > 70:  # Overbought
                confidence += 20
            elif rsi > 60:
                confidence += 15
            elif rsi > 50:
                confidence += 8
        elif bias == 'neutral':
            if 45 <= rsi <= 55:  # Neutral zone
                confidence += 12
        
        # MACD signal strength
        macd_signal = technical_analysis.get('macd_signal', 'neutral')
        if (bias == 'bullish' and macd_signal == 'bullish') or \
           (bias == 'bearish' and macd_signal == 'bearish'):
            confidence += 15
        elif macd_signal == 'neutral' and bias == 'neutral':
            confidence += 10
        
        # Trend strength with momentum factor
        trend_strength = abs(technical_analysis.get('price_change_pct', 0))
        momentum_score = technical_analysis.get('momentum_score', 0)
        
        if trend_strength > 5:
            confidence += 15
        elif trend_strength > 3:
            confidence += 12
        elif trend_strength > 1.5:
            confidence += 6
        
        # Volume confirmation
        volume_ratio = technical_analysis.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:  # High volume confirmation
            confidence += 10
        elif volume_ratio > 1.2:
            confidence += 5
        
        # Moving average alignment
        price = technical_analysis['current_price']
        sma_20 = technical_analysis['sma_20']
        sma_50 = technical_analysis['sma_50']
        
        if bias == 'bullish' and price > sma_20 > sma_50:
            confidence += 10
        elif bias == 'bearish' and price < sma_20 < sma_50:
            confidence += 10
        
        # Time decay factor
        if time_to_expiry < 0.1:  # Less than ~5 weeks
            confidence -= 10
        
        # Volume confirmation
        if technical_analysis['volume_trend'] > 1.2:
            confidence += 5
        
        return min(max(confidence, 20), 90)  # Cap between 20-90%
    
    def _calculate_pop(self, spot, strike, vol, time, option_type):
        """Calculate probability of profit"""
        try:
            # Simplified POP calculation
            if option_type == 'call':
                # Probability that spot > breakeven at expiry
                distance = (strike - spot) / spot
                pop = 50 - (distance * 100)
            else:
                # Probability that spot < breakeven at expiry
                distance = (spot - strike) / spot
                pop = 50 - (distance * 100)
            
            # Adjust for time and volatility
            vol_adjustment = vol * 100 * np.sqrt(time)
            pop += vol_adjustment * 0.3
            
            return min(max(pop, 10), 90)
        except:
            return 50
    
    def _calculate_spread_pop(self, spot, lower_strike, upper_strike, vol, time):
        """Calculate probability of profit for spreads"""
        try:
            spread_width = upper_strike - lower_strike
            spot_position = (spot - lower_strike) / spread_width
            
            # Base POP around 60% for spreads near ATM
            pop = 60 - abs(spot_position - 0.5) * 40
            
            return min(max(pop, 30), 80)
        except:
            return 55
    
    def _assess_market_conditions(self, technical_analysis, implied_vol):
        """Assess overall market conditions"""
        conditions = []
        
        vol_level = implied_vol * 100
        if vol_level > 30:
            conditions.append("High Volatility Environment")
        elif vol_level < 15:
            conditions.append("Low Volatility Environment")
        else:
            conditions.append("Normal Volatility Environment")
        
        if technical_analysis['rsi'] > 70:
            conditions.append("Overbought Conditions")
        elif technical_analysis['rsi'] < 30:
            conditions.append("Oversold Conditions")
        
        if abs(technical_analysis['price_change_pct']) > 3:
            conditions.append("High Momentum")
        
        if technical_analysis['volume_trend'] > 1.3:
            conditions.append("Above Average Volume")
        
        return conditions
    
    def _generate_fallback_recommendation(self, commodity, risk_tolerance):
        """Generate basic recommendation when data is unavailable"""
        return {
            'commodity': commodity,
            'current_price': 'Data Unavailable',
            'market_bias': 'neutral',
            'technical_summary': {'note': 'Limited data available'},
            'implied_volatility': 25.0,
            'recommendations': [{
                'strategy_name': 'Data Limited - Consult Live Markets',
                'action': 'Monitor market conditions',
                'option_type': 'Analysis Required',
                'risk_level': 'Unknown',
                'confidence_score': 0,
                'reasoning': f'Insufficient data for {commodity}. Please check live market data sources.'
            }],
            'market_conditions': ['Data Unavailable']
        }