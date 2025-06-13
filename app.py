"""
Main entry point for the Indian Commodities Options Trading Platform
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import pytz
from utils.indian_market_utils import is_market_open, get_market_status, get_indian_time
from utils.data_fetcher import get_commodity_data, get_indian_commodities
from utils.logger import logger

# Page configuration
st.set_page_config(
    page_title="Indian Commodities Options Trading Platform",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open('static/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_market_data():
    """Load market data with caching"""
    try:
        commodities = get_indian_commodities()
        data = {}
        for name, symbol in commodities.items():
            commodity_data = get_commodity_data(symbol, period='1d')
            if commodity_data is not None and not commodity_data.empty:
                data[name] = commodity_data
        return data
    except Exception as e:
        logger.error(f"Error loading market data: {str(e)}")
        return None

def main():
    try:
        # Header Banner with Modern Design
        st.markdown("""
            <div class="header-banner" style="background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                url('https://images.pexels.com/photos/374074/pexels-photo-374074.jpeg');">
                <h1>🏠 Indian Commodities Options Trading Platform</h1>
                <p>Real-time analysis and recommendations for MCX & NCDEX markets</p>
            </div>
        """, unsafe_allow_html=True)

        # Market Status Section
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            indian_time = get_indian_time()
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Market Time</h3>
                    <p style="font-size: 1.2em; font-weight: bold;">{indian_time.strftime('%Y-%m-%d %H:%M:%S')} IST</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            market_open = is_market_open()
            status_class = "market-open" if market_open else "market-closed"
            market_status = get_market_status()
            st.markdown(f"""
                <div class="market-status {status_class}">
                    {market_status}
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

        # Market Overview
        st.markdown("### 📈 Market Overview")
        
        with st.spinner("Loading market data..."):
            market_data = load_market_data()
            
            if market_data:
                # Create metrics grid
                cols = st.columns(len(market_data))
                for idx, (name, data) in enumerate(market_data.items()):
                    with cols[idx]:
                        current_price = data['Close'].iloc[-1]
                        prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                        change = current_price - prev_close
                        change_pct = (change / prev_close) * 100
                        
                        # Color coding based on price change
                        color = "#28a745" if change >= 0 else "#dc3545"
                        
                        st.markdown(f"""
                            <div class="metric-card" style="border-left: 4px solid {color}">
                                <h4>{name}</h4>
                                <p style="font-size: 1.4em; margin: 0;">₹{current_price:.2f}</p>
                                <p style="color: {color};">{change_pct:+.2f}%</p>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("Unable to load market data. Please try again later.")

        # Platform Features
        st.markdown("### 🧭 Platform Features")
        
        # Feature grid with modern cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <h4>📊 Market Analysis</h4>
                    <ul>
                        <li>Real-time commodity prices</li>
                        <li>Historical charts and analysis</li>
                        <li>Volume and open interest data</li>
                    </ul>
                </div>
                
                <div class="metric-card">
                    <h4>⚡ Options Trading</h4>
                    <ul>
                        <li>Live options chain</li>
                        <li>Greeks calculation</li>
                        <li>Implied volatility analysis</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
                <div class="metric-card">
                    <h4>🎯 Smart Recommendations</h4>
                    <ul>
                        <li>AI-powered trading signals</li>
                        <li>Technical analysis insights</li>
                        <li>Market sentiment analysis</li>
                    </ul>
                </div>
                
                <div class="metric-card">
                    <h4>🔍 Advanced Screener</h4>
                    <ul>
                        <li>Custom screening criteria</li>
                        <li>Strategy-based filters</li>
                        <li>Real-time alerts</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
                <div class="metric-card">
                    <h4>💼 Portfolio Management</h4>
                    <ul>
                        <li>Position tracking</li>
                        <li>P&L monitoring</li>
                        <li>Performance analytics</li>
                    </ul>
                </div>
                
                <div class="metric-card">
                    <h4>⚠️ Risk Management</h4>
                    <ul>
                        <li>Portfolio risk metrics</li>
                        <li>Value at Risk (VaR)</li>
                        <li>Stress testing scenarios</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        # Educational Resources
        with st.expander("📚 Options Trading Basics"):
            st.markdown("""
                <div class="metric-card">
                    <h4>Key Concepts for Indian Commodity Options</h4>
                    
                    <h5>1. Contract Specifications</h5>
                    <ul>
                        <li>MCX Gold options: 1 kg lot size</li>
                        <li>MCX Silver options: 30 kg lot size</li>
                        <li>MCX Crude oil: 100 barrels lot size</li>
                    </ul>
                    
                    <h5>2. Trading Hours</h5>
                    <ul>
                        <li>Morning session: 9:00 AM to 5:00 PM</li>
                        <li>Evening session: 5:00 PM to 11:30 PM</li>
                    </ul>
                    
                    <h5>3. Popular Strategies</h5>
                    <ul>
                        <li><strong>Bull Call Spread:</strong> Buy ATM call, sell OTM call</li>
                        <li><strong>Bear Put Spread:</strong> Buy ATM put, sell OTM put</li>
                        <li><strong>Long Straddle:</strong> Buy ATM call and put</li>
                        <li><strong>Iron Condor:</strong> Sell ATM straddle, buy protective wings</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        # Disclaimer
        st.markdown("---")
        st.markdown("""
            <div class="alert alert-warning">
                <strong>Disclaimer:</strong> This platform is for educational and informational purposes only. 
                Trading in commodities and options involves substantial risk of loss. 
                Please consult with a qualified financial advisor before making any investment decisions.
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"Error in main app: {str(e)}")
        st.error("An unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    main()
