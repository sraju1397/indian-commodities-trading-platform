import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from io import StringIO
import time
from utils.data_fetcher import get_commodity_data
from utils.simple_ai_engine import simple_ai_engine
from utils.indian_market_utils import get_indian_time, is_market_open
from config import Config

st.set_page_config(page_title="AI Trading Signals", page_icon="🎯", layout="wide")

st.title("🎯 MCX Trading Signals & Alerts")
st.markdown("### Technical alerts and AI-powered recommendations in one place")

# Authentication Status
api_key = Config.KITE_API_KEY or "caym8d0xr9e2xnh0"
access_token = st.session_state.get('kite_access_token') or Config.KITE_ACCESS_TOKEN or "Z8F0DHq2CWz4T7OJnfCmmOT40HwLm70V"

if access_token and access_token != "your_access_token_here":
    st.success("✅ AI analysis powered by live MCX data")
else:
    st.warning("⚠️ Connect to Kite for live data analysis")

# Market Status
current_time = get_indian_time()
market_status = "🟢 OPEN" if is_market_open() else "🔴 CLOSED"
st.info(f"MCX Market Status: {market_status} | IST: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Control tabs
tab1, tab2 = st.tabs(["📈 Technical Alerts", "🤖 AI Recommendations"])

# Auto-refresh control
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)

def fetch_live_mcx_price(commodity_contract, headers):
    """Fetch live MCX price for a commodity"""
    try:
        # Get instrument token from MCX instruments
        instruments_response = requests.get('https://api.kite.trade/instruments', headers=headers)
        if instruments_response.status_code == 200:
            instruments_df = pd.read_csv(StringIO(instruments_response.text))
            mcx_instruments = instruments_df[instruments_df['exchange'] == 'MCX']
            
            contract_data = mcx_instruments[
                (mcx_instruments['tradingsymbol'] == commodity_contract) & 
                (mcx_instruments['instrument_type'] == 'FUT')
            ]
            
            if not contract_data.empty:
                token = contract_data.iloc[0]['instrument_token']
                quote_response = requests.get(f'https://api.kite.trade/quote/ltp?i=MCX:{token}', headers=headers)
                if quote_response.status_code == 200:
                    quote_data = quote_response.json()
                    if 'data' in quote_data and f'MCX:{token}' in quote_data['data']:
                        return quote_data['data'][f'MCX:{token}']['last_price']
    except Exception as e:
        st.error(f"Error fetching live price: {str(e)}")
    return None

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 50

def generate_technical_alerts():
    """Generate basic technical alerts based on price action"""
    alerts = []
    
    commodities = {
        'Crude Oil': {'contract': 'CRUDEOIL25JUNFUT', 'symbol': 'CL=F'},
        'Natural Gas': {'contract': 'NATURALGAS25JUNFUT', 'symbol': 'NG=F'}
    }
    
    headers = {'Authorization': f'token {api_key}:{access_token}'}
    
    for commodity, details in commodities.items():
        live_price = fetch_live_mcx_price(details['contract'], headers)
        historical_data = get_commodity_data(details['symbol'], period='1mo', interval='1d')
        
        if historical_data is not None and not historical_data.empty and live_price:
            # Calculate technical indicators
            rsi = calculate_rsi(historical_data['Close'])
            
            # Moving averages
            sma_20 = historical_data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = historical_data['Close'].rolling(50).mean().iloc[-1] if len(historical_data) >= 50 else sma_20
            
            # Generate alerts based on conditions
            alert_type = "NEUTRAL"
            alert_message = ""
            
            if rsi < 30:
                alert_type = "BUY"
                alert_message = f"RSI Oversold: {rsi:.1f}"
            elif rsi > 70:
                alert_type = "SELL" 
                alert_message = f"RSI Overbought: {rsi:.1f}"
            elif live_price > sma_20 > sma_50:
                alert_type = "BUY"
                alert_message = "Bullish trend: Price above moving averages"
            elif live_price < sma_20 < sma_50:
                alert_type = "SELL"
                alert_message = "Bearish trend: Price below moving averages"
            else:
                alert_message = "No clear signal"
            
            # Calculate volatility for position sizing
            returns = historical_data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            
            alerts.append({
                'Time': current_time.strftime('%H:%M:%S'),
                'Commodity': commodity,
                'Current Price': f"₹{live_price:.1f}",
                'Alert Type': alert_type,
                'Signal': alert_message,
                'RSI': f"{rsi:.1f}",
                'SMA 20': f"₹{sma_20:.1f}",
                'SMA 50': f"₹{sma_50:.1f}",
                'Volatility': f"{volatility:.1f}%"
            })
    
    return alerts

def generate_advanced_recommendations():
    """Generate AI-powered trading recommendations"""
    recommendations = []
    
    # MCX commodity contracts (June 2025)
    commodities = {
        'Crude Oil': {'contract': 'CRUDEOIL25JUNFUT', 'symbol': 'CL=F'},
        'Natural Gas': {'contract': 'NATURALGAS25JUNFUT', 'symbol': 'NG=F'}
    }
    
    headers = {'Authorization': f'token {api_key}:{access_token}'}
    
    for commodity, details in commodities.items():
        # Get live price from MCX
        live_price = fetch_live_mcx_price(details['contract'], headers)
        
        # Get historical data for AI analysis
        historical_data = get_commodity_data(details['symbol'], period='3mo', interval='1d')
        
        if historical_data is not None and not historical_data.empty and live_price:
            # Use live price as current price
            historical_data.loc[historical_data.index[-1], 'Close'] = live_price
            
            # Generate AI market signals
            market_signals = simple_ai_engine.generate_trading_signals(historical_data)
            
            # Generate specific options recommendations
            ai_recommendations = simple_ai_engine.generate_options_recommendations(
                commodity, live_price, market_signals
            )
            
            for rec in ai_recommendations:
                # Format entry zone properly
                entry_zone_text = "Market Price"
                if rec.get('entry_zone') and isinstance(rec['entry_zone'], list):
                    entry_zone_text = f"₹{rec['entry_zone'][0]:.1f} - ₹{rec['entry_zone'][1]:.1f}"
                
                recommendations.append({
                    'Time': current_time.strftime('%H:%M:%S'),
                    'Commodity': commodity,
                    'Signal': market_signals['signal'],
                    'Option': rec['instrument'],
                    'Type': rec['type'],
                    'Strike': rec['strike'],
                    'Current Price': f"₹{live_price:.1f}",
                    'Entry Zone': entry_zone_text,
                    'Target 1': f"₹{rec['targets'][0]:.1f}" if rec['targets'] else "N/A",
                    'Target 2': f"₹{rec['targets'][1]:.1f}" if len(rec['targets']) > 1 else "N/A",
                    'Stop Loss': f"₹{rec['stop_loss']:.1f}" if rec['stop_loss'] else "N/A",
                    'Confidence': rec['confidence'],
                    'Risk Level': rec['risk_level'],
                    'Max Loss': rec['max_loss'],
                    'Strategy': rec['strategy'],
                    'Reasoning': rec['reasoning']
                })
    
    return recommendations

# Technical Alerts Tab
with tab1:
    st.subheader("📈 Technical Analysis Alerts")
    
    try:
        technical_alerts = generate_technical_alerts()
        
        if technical_alerts:
            # Display alerts as colored cards
            for alert in technical_alerts:
                if alert['Alert Type'] == 'BUY':
                    st.success(f"**{alert['Commodity']}** - {alert['Signal']} | Current: {alert['Current Price']} | RSI: {alert['RSI']}")
                elif alert['Alert Type'] == 'SELL':
                    st.error(f"**{alert['Commodity']}** - {alert['Signal']} | Current: {alert['Current Price']} | RSI: {alert['RSI']}")
                else:
                    st.info(f"**{alert['Commodity']}** - {alert['Signal']} | Current: {alert['Current Price']} | RSI: {alert['RSI']}")
            
            # Technical alerts table
            st.subheader("Technical Indicators Summary")
            df_tech = pd.DataFrame(technical_alerts)
            st.dataframe(df_tech, use_container_width=True)
            
        else:
            st.warning("No technical alerts generated. Please check your authentication.")
            
    except Exception as e:
        st.error(f"Error generating technical alerts: {str(e)}")

# AI Recommendations Tab  
with tab2:
    st.subheader("🤖 AI-Powered Options Recommendations")
    
    try:
        recommendations = generate_advanced_recommendations()
        
        if recommendations:
            # Display recommendations in an enhanced format
            for i, rec in enumerate(recommendations):
                # Color coding based on confidence
                if rec['Confidence'] == 'HIGH':
                    border_color = "#28a745"  # Green
                elif rec['Confidence'] == 'MEDIUM':
                    border_color = "#ffc107"  # Yellow
                else:
                    border_color = "#dc3545"  # Red
                
                # Create recommendation card
                st.markdown(f"""
                <div style="border-left: 5px solid {border_color}; padding: 15px; margin: 10px 0; background-color: #f8f9fa; border-radius: 5px;">
                    <h4 style="margin: 0; color: #333;">
                        {rec['Signal']} {rec['Option']} | Confidence: {rec['Confidence']}
                    </h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 10px;">
                        <div>
                            <strong>Entry Details:</strong><br>
                            Current: {rec['Current Price']}<br>
                            Entry Zone: {rec['Entry Zone']}<br>
                            Risk Level: {rec['Risk Level']}
                        </div>
                        <div>
                            <strong>Targets & Stop Loss:</strong><br>
                            Target 1: {rec['Target 1']}<br>
                            Target 2: {rec['Target 2']}<br>
                            Stop Loss: {rec['Stop Loss']}
                        </div>
                        <div>
                            <strong>Risk Management:</strong><br>
                            Max Loss: {rec['Max Loss']}<br>
                            Strategy: {rec['Strategy'][:30]}...
                        </div>
                    </div>
                    <div style="margin-top: 10px; padding: 8px; background-color: #e9ecef; border-radius: 3px;">
                        <strong>AI Analysis:</strong> {rec['Reasoning']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Summary table
            st.subheader("📊 Quick Reference Table")
            df = pd.DataFrame(recommendations)
            display_columns = ['Time', 'Commodity', 'Signal', 'Option', 'Current Price', 
                              'Target 1', 'Stop Loss', 'Confidence', 'Risk Level']
            st.dataframe(df[display_columns], use_container_width=True)
            
            # Risk Management Guidelines
            st.subheader("⚠️ Risk Management Rules")
            st.info("""
            **Essential Trading Rules:**
            • Never risk more than 2-3% of your capital per trade
            • Always use stop losses - exit if underlying moves against you by 1.5x ATR
            • Book 50% profits at first target, trail the rest
            • Exit options 1 week before expiry if not profitable
            • Maximum 2-3 positions per commodity simultaneously
            • Monitor market news and events that could impact prices
            """)
            
            # Performance metrics (if historical data available)
            st.subheader("📈 Signal Performance")
            col1, col2, col3, col4 = st.columns(4)
            
            high_conf_signals = len([r for r in recommendations if r['Confidence'] == 'HIGH'])
            medium_conf_signals = len([r for r in recommendations if r['Confidence'] == 'MEDIUM'])
            
            with col1:
                st.metric("Total Signals", len(recommendations))
            with col2:
                st.metric("High Confidence", high_conf_signals)
            with col3:
                st.metric("Medium Confidence", medium_conf_signals)
            with col4:
                st.metric("Market Status", "Active" if is_market_open() else "Closed")
        
        else:
            st.warning("No AI recommendations generated. Please check your authentication.")
            
    except Exception as e:
        st.error(f"Error generating AI recommendations: {str(e)}")

# Footer
st.markdown("---")
st.caption(f"Last updated: {current_time.strftime('%Y-%m-%d %H:%M:%S')} IST | Powered by AI Technical Analysis")

# Auto-refresh mechanism
if auto_refresh:
    time.sleep(30)
    st.rerun()