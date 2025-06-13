import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from io import StringIO
from datetime import datetime, timedelta
from utils.data_fetcher import get_commodity_data, get_indian_commodities
from utils.options_calculator import black_scholes_price, calculate_greeks
from utils.indian_market_utils import get_indian_time, is_market_open
from utils.options_recommendation_engine import OptionsRecommendationEngine
from config import Config

st.set_page_config(page_title="Recommendations", page_icon="🎯", layout="wide")

st.title("🎯 Trading Recommendations")
st.markdown("### MCX options recommendations powered by live data analysis")

# MCX Authentication Status
api_key = Config.KITE_API_KEY or "caym8d0xr9e2xnh0"
access_token = st.session_state.get('kite_access_token') or Config.KITE_ACCESS_TOKEN or "Z8F0DHq2CWz4T7OJnfCmmOT40HwLm70V"

if access_token and access_token != "your_access_token_here":
    st.success("✅ Live MCX data connected for authentic recommendations")
else:
    st.warning("⚠️ Kite Connect authentication required for live MCX recommendations.")

# Sidebar controls
st.sidebar.header("Recommendation Filters")

# Commodity selection
indian_commodities = get_indian_commodities()
default_commodities = ['Gold', 'Silver', 'Crude Oil', 'Natural Gas', 'Copper']
selected_commodities = st.sidebar.multiselect(
    "Select Commodities",
    options=list(indian_commodities.keys()),
    default=default_commodities
)

# Signal type filter
signal_types = st.sidebar.multiselect(
    "Signal Types",
    options=["Technical", "Volatility", "Options Flow", "Momentum"],
    default=["Technical", "Volatility"]
)

# Risk level filter
risk_level = st.sidebar.selectbox(
    "Risk Level",
    options=["Conservative", "Moderate", "Aggressive"],
    index=1
)

# Time horizon
time_horizon = st.sidebar.selectbox(
    "Time Horizon",
    options=["Intraday", "Short-term (1-7 days)", "Medium-term (1-4 weeks)"],
    index=1
)

def calculate_technical_signals(data):
    """Calculate technical analysis signals"""
    signals = {}
    
    if data is None or data.empty:
        return signals
    
    # Moving averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Histogram'] = data['MACD'] - data['Signal']
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Std'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
    data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
    
    # Stochastic Oscillator
    data['Lowest_Low'] = data['Low'].rolling(window=14).min()
    data['Highest_High'] = data['High'].rolling(window=14).max()
    data['%K'] = 100 * ((data['Close'] - data['Lowest_Low']) / (data['Highest_High'] - data['Lowest_Low']))
    data['%D'] = data['%K'].rolling(window=3).mean()
    
    # Average True Range (ATR) for volatility
    data['TR1'] = data['High'] - data['Low']
    data['TR2'] = abs(data['High'] - data['Close'].shift(1))
    data['TR3'] = abs(data['Low'] - data['Close'].shift(1))
    data['TR'] = data[['TR1', 'TR2', 'TR3']].max(axis=1)
    data['ATR'] = data['TR'].rolling(window=14).mean()
    
    # Williams %R
    data['Williams_%R'] = -100 * ((data['Highest_High'] - data['Close']) / (data['Highest_High'] - data['Lowest_Low']))
    
    # Volume indicators
    data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
    
    # Commodity Channel Index (CCI)
    data['TP'] = (data['High'] + data['Low'] + data['Close']) / 3  # Typical Price
    data['TP_SMA'] = data['TP'].rolling(window=20).mean()
    data['Mean_Dev'] = data['TP'].rolling(window=20).apply(lambda x: abs(x - x.mean()).mean())
    data['CCI'] = (data['TP'] - data['TP_SMA']) / (0.015 * data['Mean_Dev'])
    
    # Get latest values
    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else latest
    
    # Generate signals
    signals['price'] = latest['Close']
    signals['change_pct'] = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
    
    # Moving Average Signal
    if latest['Close'] > latest['SMA_20'] and latest['SMA_20'] > latest['SMA_50']:
        signals['ma_signal'] = "BULLISH"
        signals['ma_strength'] = 3
    elif latest['Close'] < latest['SMA_20'] and latest['SMA_20'] < latest['SMA_50']:
        signals['ma_signal'] = "BEARISH"
        signals['ma_strength'] = 3
    else:
        signals['ma_signal'] = "NEUTRAL"
        signals['ma_strength'] = 1
    
    # RSI Signal
    if latest['RSI'] < 30:
        signals['rsi_signal'] = "OVERSOLD"
        signals['rsi_strength'] = 3
    elif latest['RSI'] > 70:
        signals['rsi_signal'] = "OVERBOUGHT"
        signals['rsi_strength'] = 3
    else:
        signals['rsi_signal'] = "NEUTRAL"
        signals['rsi_strength'] = 1
    
    # MACD Signal
    if latest['MACD'] > latest['Signal'] and prev['MACD'] <= prev['Signal']:
        signals['macd_signal'] = "BULLISH CROSSOVER"
        signals['macd_strength'] = 3
    elif latest['MACD'] < latest['Signal'] and prev['MACD'] >= prev['Signal']:
        signals['macd_signal'] = "BEARISH CROSSOVER"
        signals['macd_strength'] = 3
    else:
        signals['macd_signal'] = "NEUTRAL"
        signals['macd_strength'] = 1
    
    # Bollinger Bands Signal
    if latest['Close'] < latest['BB_Lower']:
        signals['bb_signal'] = "OVERSOLD"
        signals['bb_strength'] = 2
    elif latest['Close'] > latest['BB_Upper']:
        signals['bb_signal'] = "OVERBOUGHT"
        signals['bb_strength'] = 2
    else:
        signals['bb_signal'] = "NEUTRAL"
        signals['bb_strength'] = 1
    
    # Overall sentiment
    bullish_signals = sum([1 for s in [signals['ma_signal'], signals['rsi_signal'], 
                                      signals['macd_signal'], signals['bb_signal']] 
                          if 'BULLISH' in s or 'OVERSOLD' in s])
    bearish_signals = sum([1 for s in [signals['ma_signal'], signals['rsi_signal'], 
                                      signals['macd_signal'], signals['bb_signal']] 
                          if 'BEARISH' in s or 'OVERBOUGHT' in s])
    
    if bullish_signals > bearish_signals:
        signals['overall_sentiment'] = "BULLISH"
    elif bearish_signals > bullish_signals:
        signals['overall_sentiment'] = "BEARISH"
    else:
        signals['overall_sentiment'] = "NEUTRAL"
    
    signals['confidence'] = max(bullish_signals, bearish_signals) / 4.0
    
    return signals

def generate_options_recommendations(commodity, signals, current_price):
    """Generate options trading recommendations based on signals"""
    recommendations = []
    
    if signals['overall_sentiment'] == "BULLISH":
        # Bull Call Spread
        buy_strike = round(current_price * 1.02 / 50) * 50
        sell_strike = round(current_price * 1.08 / 50) * 50
        
        recommendations.append({
            'strategy': 'Bull Call Spread',
            'direction': 'BULLISH',
            'confidence': signals['confidence'],
            'action': f"Buy {buy_strike} Call, Sell {sell_strike} Call",
            'rationale': f"Technical indicators suggest upward momentum. {signals['ma_signal']} MA signal with {signals['rsi_signal']} RSI.",
            'max_profit': f"₹{sell_strike - buy_strike - 50:.0f} (estimated)",
            'max_loss': "₹50 (estimated premium)",
            'target': f"₹{sell_strike}",
            'stop_loss': f"₹{current_price * 0.98:.0f}"
        })
        
        # Long Call (aggressive)
        if risk_level == "Aggressive":
            atm_strike = round(current_price / 50) * 50
            recommendations.append({
                'strategy': 'Long Call',
                'direction': 'BULLISH',
                'confidence': signals['confidence'],
                'action': f"Buy {atm_strike} Call",
                'rationale': f"Strong bullish signals with {signals['macd_signal']} MACD and favorable momentum.",
                'max_profit': "Unlimited",
                'max_loss': "Premium paid",
                'target': f"₹{current_price * 1.10:.0f}",
                'stop_loss': f"₹{current_price * 0.95:.0f}"
            })
    
    elif signals['overall_sentiment'] == "BEARISH":
        # Bear Put Spread
        buy_strike = round(current_price * 0.98 / 50) * 50
        sell_strike = round(current_price * 0.92 / 50) * 50
        
        recommendations.append({
            'strategy': 'Bear Put Spread',
            'direction': 'BEARISH',
            'confidence': signals['confidence'],
            'action': f"Buy {buy_strike} Put, Sell {sell_strike} Put",
            'rationale': f"Technical indicators suggest downward pressure. {signals['ma_signal']} MA signal.",
            'max_profit': f"₹{buy_strike - sell_strike - 30:.0f} (estimated)",
            'max_loss': "₹30 (estimated premium)",
            'target': f"₹{sell_strike}",
            'stop_loss': f"₹{current_price * 1.02:.0f}"
        })
    
    else:  # NEUTRAL
        # Iron Condor or Straddle based on volatility
        atm_strike = round(current_price / 50) * 50
        
        if signals.get('bb_signal') == "NEUTRAL":
            recommendations.append({
                'strategy': 'Iron Condor',
                'direction': 'NEUTRAL',
                'confidence': signals['confidence'],
                'action': f"Sell {atm_strike-50} Put, Buy {atm_strike-100} Put, Sell {atm_strike+50} Call, Buy {atm_strike+100} Call",
                'rationale': "Market showing sideways movement. Profit from time decay in range-bound market.",
                'max_profit': "₹40 (estimated net credit)",
                'max_loss': "₹10 (estimated)",
                'target': f"₹{atm_strike-50} - ₹{atm_strike+50}",
                'stop_loss': "Early exit if breakout occurs"
            })
    
    return recommendations

# Main content
st.subheader("📊 Live Market Signals")

if not selected_commodities:
    st.warning("Please select at least one commodity to view recommendations.")
else:
    # Create tabs for different types of recommendations
    tab1, tab2, tab3 = st.tabs(["🔄 Active Signals", "📈 Strategy Recommendations", "📊 Market Analysis"])
    
    with tab1:
        st.markdown("### Current Trading Signals")
        
        signals_data = []
        
        for commodity in selected_commodities:
            try:
                symbol = indian_commodities[commodity]
                data = get_commodity_data(symbol, period='3mo')
                
                if data is not None and not data.empty:
                    signals = calculate_technical_signals(data)
                    
                    # Determine signal strength color
                    if signals['overall_sentiment'] == "BULLISH":
                        sentiment_color = "🟢"
                    elif signals['overall_sentiment'] == "BEARISH":
                        sentiment_color = "🔴"
                    else:
                        sentiment_color = "🟡"
                    
                    signals_data.append({
                        'Commodity': commodity,
                        'Price': f"₹{signals['price']:.2f}",
                        'Change %': f"{signals['change_pct']:+.2f}%",
                        'Sentiment': f"{sentiment_color} {signals['overall_sentiment']}",
                        'Confidence': f"{signals['confidence']:.0%}",
                        'MA Signal': signals['ma_signal'],
                        'RSI Signal': signals['rsi_signal'],
                        'MACD Signal': signals['macd_signal'],
                        'BB Signal': signals['bb_signal']
                    })
            except Exception as e:
                st.error(f"Error processing {commodity}: {str(e)}")
        
        if signals_data:
            df_signals = pd.DataFrame(signals_data)
            st.dataframe(df_signals, use_container_width=True)
        
        # Real-time alerts
        st.subheader("🚨 Real-time Alerts")
        
        alert_col1, alert_col2 = st.columns(2)
        
        with alert_col1:
            st.info("**Active Alerts**")
            for commodity in selected_commodities[:2]:
                try:
                    symbol = indian_commodities[commodity]
                    data = get_commodity_data(symbol, period='1d')
                    if data is not None and not data.empty:
                        signals = calculate_technical_signals(data)
                        if signals['confidence'] > 0.6:
                            st.success(f"🎯 **{commodity}**: {signals['overall_sentiment']} signal with {signals['confidence']:.0%} confidence")
                except:
                    continue
        
        with alert_col2:
            st.info("**Market Status**")
            market_open = is_market_open()
            indian_time = get_indian_time()
            
            if market_open:
                st.success(f"🟢 Markets are OPEN | {indian_time.strftime('%H:%M:%S IST')}")
            else:
                st.error(f"🔴 Markets are CLOSED | {indian_time.strftime('%H:%M:%S IST')}")
    
    with tab2:
        st.markdown("### 🎯 Intelligent Options Recommendations")
        st.markdown("*Specific strike prices, risk levels, and confidence scores*")
        
        # Initialize recommendation engine
        rec_engine = OptionsRecommendationEngine()
        
        # Risk tolerance selector
        col1, col2 = st.columns([1, 2])
        with col1:
            risk_tolerance = st.selectbox(
                "Risk Tolerance",
                options=['conservative', 'moderate', 'aggressive'],
                index=1,
                help="Conservative: Lower risk spreads, Moderate: Balanced strategies, Aggressive: Higher risk/reward"
            )
        
        for commodity in selected_commodities:
            try:
                symbol = indian_commodities[commodity]
                
                # Get intelligent recommendations
                analysis = rec_engine.analyze_commodity_and_recommend(commodity, symbol, risk_tolerance)
                
                st.subheader(f"📊 {commodity} Analysis & Recommendations")
                
                # Market overview
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if isinstance(analysis['current_price'], (int, float)):
                        st.metric("Current Price", f"₹{analysis['current_price']:,.2f}")
                    else:
                        st.metric("Current Price", "Data Limited")
                
                with col2:
                    bias_color = "🟢" if analysis['market_bias'] == 'bullish' else "🔴" if analysis['market_bias'] == 'bearish' else "🟡"
                    st.metric("Market Bias", f"{bias_color} {analysis['market_bias'].title()}")
                
                with col3:
                    st.metric("Implied Volatility", f"{analysis['implied_volatility']:.1f}%")
                
                with col4:
                    if analysis['market_conditions']:
                        st.metric("Market Condition", analysis['market_conditions'][0])
                
                # Specific recommendations
                if analysis['recommendations']:
                    st.markdown("#### 💡 Specific Trading Suggestions")
                    
                    for i, rec in enumerate(analysis['recommendations']):
                        # Skip fallback recommendations
                        if rec['confidence_score'] == 0:
                            continue
                            
                        # Color coding for risk levels
                        risk_colors = {
                            'Low': '🟢',
                            'Medium': '🟡', 
                            'High': '🔴',
                            'Very High': '🔴🔴'
                        }
                        
                        risk_color = risk_colors.get(rec['risk_level'], '🟡')
                        
                        with st.expander(f"{rec['strategy_name']} | {risk_color} {rec['risk_level']} Risk | {rec['confidence_score']:.0f}% Confidence"):
                            
                            # Main recommendation details
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**📋 Trade Details:**")
                                st.markdown(f"**Action:** {rec['action']}")
                                st.markdown(f"**Strike:** ₹{rec['strike_price']}")
                                st.markdown(f"**Expiry:** {rec['expiry']}")
                                st.markdown(f"**Premium:** ₹{rec['premium']:.2f}")
                                
                                if 'quantity_suggested' in rec:
                                    st.markdown(f"**Suggested Qty:** {rec['quantity_suggested']} lot(s)")
                                
                                if 'breakeven' in rec:
                                    if isinstance(rec['breakeven'], (int, float)):
                                        st.markdown(f"**Breakeven:** ₹{rec['breakeven']:.2f}")
                                    else:
                                        st.markdown(f"**Breakeven:** {rec['breakeven']}")
                            
                            with col2:
                                st.markdown("**💰 Risk & Reward:**")
                                
                                if isinstance(rec['max_loss'], (int, float)):
                                    st.metric("Max Loss", f"₹{rec['max_loss']:.2f}")
                                else:
                                    st.metric("Max Loss", rec['max_loss'])
                                
                                if isinstance(rec['target_profit'], (int, float)):
                                    st.metric("Target Profit", f"₹{rec['target_profit']:.2f}")
                                else:
                                    st.metric("Target Profit", rec['target_profit'])
                                
                                if 'probability_of_profit' in rec:
                                    st.metric("Profit Probability", f"{rec['probability_of_profit']:.0f}%")
                            
                            # Price targets and stops
                            st.markdown("**🎯 Price Targets:**")
                            target_col1, target_col2 = st.columns(2)
                            with target_col1:
                                if isinstance(rec['target_price'], (int, float)):
                                    st.markdown(f"**Target Price:** ₹{rec['target_price']:.2f}")
                                else:
                                    st.markdown(f"**Target Price:** {rec['target_price']}")
                            
                            with target_col2:
                                if isinstance(rec['stop_loss'], (int, float)):
                                    st.markdown(f"**Stop Loss:** ₹{rec['stop_loss']:.2f}")
                                else:
                                    st.markdown(f"**Stop Loss:** {rec['stop_loss']}")
                            
                            # Greeks (if available)
                            if 'delta' in rec and rec['delta']:
                                st.markdown("**📊 Greeks:**")
                                greeks_col1, greeks_col2, greeks_col3 = st.columns(3)
                                with greeks_col1:
                                    st.metric("Delta", f"{rec['delta']:.3f}")
                                with greeks_col2:
                                    if 'gamma' in rec:
                                        st.metric("Gamma", f"{rec['gamma']:.4f}")
                                with greeks_col3:
                                    if 'theta' in rec:
                                        st.metric("Theta", f"{rec['theta']:.3f}")
                            
                            # Reasoning
                            st.markdown("**💭 Analysis & Reasoning:**")
                            st.info(rec['reasoning'])
                            
                            # Confidence and risk assessment
                            confidence_score = rec['confidence_score']
                            if confidence_score >= 75:
                                st.success(f"✅ **High Confidence ({confidence_score}%)** - Strong technical signals support this trade")
                            elif confidence_score >= 60:
                                st.warning(f"⚠️ **Moderate Confidence ({confidence_score}%)** - Mixed signals, proceed with caution")
                            else:
                                st.info(f"ℹ️ **Lower Confidence ({confidence_score}%)** - Weak signals, consider waiting for better setup")
                
                else:
                    st.info(f"No specific recommendations available for {commodity} at current market conditions.")
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"Error generating recommendations for {commodity}: {str(e)}")
                with st.expander("Debug Info"):
                    st.write(f"Error details: {e}")
        
        # General trading tips
        with st.expander("📚 Options Trading Tips"):
            st.markdown("""
            #### Key Guidelines for Options Trading:
            
            **Risk Management:**
            - Never risk more than 2-5% of your portfolio on a single trade
            - Always define your exit strategy before entering
            - Consider time decay (Theta) especially for option buyers
            
            **Strategy Selection:**
            - **Bullish:** Long Calls, Bull Call Spreads
            - **Bearish:** Long Puts, Bear Put Spreads  
            - **Neutral:** Iron Condors, Short Straddles
            
            **Timing Considerations:**
            - High IV: Consider selling strategies
            - Low IV: Consider buying strategies
            - Time decay accelerates in final 30 days
            
            **Position Sizing:**
            - Conservative: 1-2 lots per trade
            - Moderate: 2-3 lots per trade
            - Aggressive: 3-5 lots per trade
            """)
    
    
    with tab3:
        st.markdown("### Market Analysis & Insights")
        
        # Market sentiment overview
        st.subheader("📊 Overall Market Sentiment")
        
        try:
            all_signals = []
            for commodity in selected_commodities:
                symbol = indian_commodities[commodity]
                data = get_commodity_data(symbol, period='1mo')
                if data is not None and not data.empty:
                    signals = calculate_technical_signals(data)
                    all_signals.append(signals)
            
            if all_signals:
                bullish_count = sum(1 for s in all_signals if s['overall_sentiment'] == 'BULLISH')
                bearish_count = sum(1 for s in all_signals if s['overall_sentiment'] == 'BEARISH')
                neutral_count = len(all_signals) - bullish_count - bearish_count
                
                # Market sentiment pie chart
                fig_sentiment = go.Figure(data=go.Pie(
                    labels=['Bullish', 'Bearish', 'Neutral'],
                    values=[bullish_count, bearish_count, neutral_count],
                    hole=0.3,
                    marker_colors=['#28a745', '#dc3545', '#ffc107']
                ))
                fig_sentiment.update_layout(
                    title="Market Sentiment Distribution",
                    height=400
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
                
                # Key insights
                st.subheader("🔍 Key Insights")
                
                avg_confidence = np.mean([s['confidence'] for s in all_signals])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Bullish Commodities", bullish_count)
                
                with col2:
                    st.metric("Bearish Commodities", bearish_count)
                
                with col3:
                    st.metric("Average Confidence", f"{avg_confidence:.0%}")
                
                # Market outlook
                if bullish_count > bearish_count:
                    st.success("🟢 **Market Outlook:** Generally bullish sentiment across commodities")
                elif bearish_count > bullish_count:
                    st.error("🔴 **Market Outlook:** Generally bearish sentiment across commodities")
                else:
                    st.warning("🟡 **Market Outlook:** Mixed signals, sideways market expected")
        
        except Exception as e:
            st.error(f"Error in market analysis: {str(e)}")

# Educational content
with st.expander("📚 Understanding Trading Signals"):
    st.markdown("""
    #### Signal Types Explained:
    
    **Technical Signals:**
    - **Moving Average (MA):** Trend direction based on price averages
    - **RSI:** Overbought (>70) or Oversold (<30) conditions
    - **MACD:** Momentum and trend changes
    - **Bollinger Bands:** Volatility and mean reversion signals
    
    **Confidence Levels:**
    - **High (70%+):** Multiple indicators align, strong signal
    - **Moderate (50-70%):** Some indicators align, proceed with caution
    - **Low (<50%):** Conflicting signals, avoid or wait for clarity
    
    **Risk Management:**
    - Always use stop losses
    - Position size according to risk tolerance
    - Diversify across multiple commodities
    - Consider market timing and volatility
    """)

# Auto-refresh for live data
if st.sidebar.checkbox("Auto Refresh (60s)", value=False):
    import time
    time.sleep(60)
    st.rerun()
