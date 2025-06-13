import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, time
import pytz
from utils.indian_market_utils import is_market_open, get_market_status, get_indian_time
from utils.data_fetcher import get_commodity_data

# Page configuration
st.set_page_config(
    page_title="Indian Commodities Options Trading Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Indian market styling
st.markdown("""
<style>
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28a745;
}
.profit {
    color: #28a745;
}
.loss {
    color: #dc3545;
}
.market-status {
    font-weight: bold;
    padding: 0.5rem;
    border-radius: 0.3rem;
    text-align: center;
}
.market-open {
    background-color: #d4edda;
    color: #155724;
}
.market-closed {
    background-color: #f8d7da;
    color: #721c24;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.title("🏛️ Indian Commodities Options Trading Platform")
    st.markdown("### Real-time analysis and recommendations for MCX & NCDEX markets")
    
    # Market status
    indian_time = get_indian_time()
    market_open = is_market_open()
    market_status = get_market_status()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"**Current IST Time:** {indian_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    with col2:
        status_class = "market-open" if market_open else "market-closed"
        st.markdown(f'<div class="market-status {status_class}">{market_status}</div>', 
                   unsafe_allow_html=True)
    
    with col3:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Quick market overview
    st.subheader("📈 Market Overview")
    
    # Major commodities data
    commodities = {
        'Gold': 'GC=F',
        'Silver': 'SI=F', 
        'Crude Oil': 'CL=F',
        'Natural Gas': 'NG=F',
        'Copper': 'HG=F'
    }
    
    cols = st.columns(len(commodities))
    
    try:
        for i, (name, symbol) in enumerate(commodities.items()):
            with cols[i]:
                data = get_commodity_data(symbol, period='1d')
                if data is not None and not data.empty:
                    current_price = data['Close'].iloc[-1]
                    prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    # Display metric
                    st.metric(
                        label=name,
                        value=f"₹{current_price:.2f}",
                        delta=f"{change_pct:.2f}%"
                    )
                else:
                    st.metric(
                        label=name,
                        value="N/A",
                        delta="Data unavailable"
                    )
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
    
    # Navigation guide
    st.subheader("🧭 Platform Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **📊 Market Data**
        - Real-time commodity prices
        - Historical charts and analysis
        - Volume and open interest data
        """)
        
        st.info("""
        **⚡ Options Chain**
        - Live options prices
        - Greeks calculation (Delta, Gamma, Theta, Vega)
        - Implied volatility analysis
        """)
    
    with col2:
        st.info("""
        **🎯 Recommendations**
        - AI-powered trading signals
        - Technical analysis insights
        - Market sentiment analysis
        """)
        
        st.info("""
        **🔍 Screener**
        - Options opportunity scanner
        - Custom filter criteria
        - Strategy-based screening
        """)
    
    with col3:
        st.info("""
        **💼 Portfolio**
        - Position tracking
        - P&L monitoring
        - Performance analytics
        """)
        
        st.info("""
        **⚠️ Risk Dashboard**
        - Portfolio risk metrics
        - Value at Risk (VaR)
        - Stress testing scenarios
        """)
    
    # Educational content
    with st.expander("📚 Options Trading Basics for Indian Markets"):
        st.markdown("""
        #### Key Concepts for Indian Commodity Options:
        
        **1. Contract Specifications:**
        - MCX Gold options: 1 kg lot size
        - MCX Silver options: 30 kg lot size
        - MCX Crude oil: 100 barrels lot size
        
        **2. Trading Hours:**
        - Morning session: 9:00 AM to 5:00 PM
        - Evening session: 5:00 PM to 11:30 PM
        
        **3. Settlement:**
        - Physical delivery or cash settlement
        - European style exercise
        
        **4. Margin Requirements:**
        - SPAN + Exposure margins
        - Mark-to-market settlements
        
        **5. Popular Strategies:**
        - **Bull Call Spread:** Buy ATM call, sell OTM call
        - **Bear Put Spread:** Buy ATM put, sell OTM put
        - **Long Straddle:** Buy ATM call and put
        - **Iron Condor:** Sell ATM straddle, buy protective wings
        """)
    
    # Disclaimer
    st.markdown("---")
    st.caption("""
    **Disclaimer:** This platform is for educational and informational purposes only. 
    Trading in commodities and options involves substantial risk of loss. 
    Please consult with a qualified financial advisor before making any investment decisions.
    """)

if __name__ == "__main__":
    main()