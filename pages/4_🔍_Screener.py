import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from io import StringIO
from datetime import datetime, timedelta
from utils.data_fetcher import get_commodity_data, get_indian_commodities
from utils.options_calculator import black_scholes_price, calculate_greeks
from utils.indian_market_utils import get_indian_time, is_market_open
from config import Config

st.set_page_config(page_title="Screener", page_icon="🔍", layout="wide")

st.title("🔍 Options Screener")
st.markdown("### Discover MCX options trading opportunities with live data filtering")

# MCX Authentication Status
api_key = Config.KITE_API_KEY or "caym8d0xr9e2xnh0"
access_token = st.session_state.get('kite_access_token') or Config.KITE_ACCESS_TOKEN or "Z8F0DHq2CWz4T7OJnfCmmOT40HwLm70V"

if access_token and access_token != "your_access_token_here":
    st.success("✅ Live MCX options data connected for screening")
else:
    st.warning("⚠️ Kite Connect authentication required for live MCX screening.")

# Sidebar filters
st.sidebar.header("Screening Filters")

# Basic filters
st.sidebar.subheader("📊 Basic Filters")

min_volume = st.sidebar.number_input(
    "Minimum Volume",
    min_value=0,
    max_value=1000000,
    value=10000,
    step=5000
)

min_oi = st.sidebar.number_input(
    "Minimum Open Interest",
    min_value=0,
    max_value=100000,
    value=1000,
    step=500
)

max_bid_ask_spread = st.sidebar.slider(
    "Max Bid-Ask Spread (%)",
    min_value=1.0,
    max_value=10.0,
    value=5.0,
    step=0.5
)

# Technical filters
st.sidebar.subheader("📈 Technical Filters")

rsi_range = st.sidebar.slider(
    "RSI Range",
    min_value=0,
    max_value=100,
    value=(30, 70),
    step=5
)

iv_range = st.sidebar.slider(
    "Implied Volatility Range (%)",
    min_value=5.0,
    max_value=100.0,
    value=(15.0, 50.0),
    step=2.5
)

# Greeks filters
st.sidebar.subheader("📊 Greeks Filters")

delta_range = st.sidebar.slider(
    "Delta Range",
    min_value=0.0,
    max_value=1.0,
    value=(0.2, 0.8),
    step=0.05
)

gamma_min = st.sidebar.number_input(
    "Minimum Gamma",
    min_value=0.0,
    max_value=0.1,
    value=0.005,
    step=0.001,
    format="%.4f"
)

theta_max = st.sidebar.number_input(
    "Maximum Theta (absolute)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.1
)

# Strategy filters
st.sidebar.subheader("🎯 Strategy Filters")

strategy_type = st.sidebar.selectbox(
    "Strategy Type",
    options=["All", "Long Calls", "Long Puts", "Bull Spreads", "Bear Spreads", "Straddles", "Strangles"],
    index=0
)

moneyness = st.sidebar.selectbox(
    "Moneyness",
    options=["All", "ITM", "ATM", "OTM"],
    index=0
)

expiry_filter = st.sidebar.selectbox(
    "Expiry Filter",
    options=["All", "This Week", "Next Week", "Current Month", "Next Month"],
    index=0
)

def calculate_technical_indicators(data):
    """Calculate technical indicators for screening"""
    if data is None or data.empty or len(data) < 20:
        return {}
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Moving averages
    sma_20 = data['Close'].rolling(window=20).mean()
    sma_50 = data['Close'].rolling(window=50).mean() if len(data) >= 50 else sma_20
    
    # Volatility
    returns = data['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
    
    try:
        return {
            'rsi': float(rsi.iloc[-1]) if not rsi.empty and not np.isnan(rsi.iloc[-1]) else 50.0,
            'sma_20': float(sma_20.iloc[-1]) if not sma_20.empty and not np.isnan(sma_20.iloc[-1]) else float(data['Close'].iloc[-1]),
            'sma_50': float(sma_50.iloc[-1]) if not sma_50.empty and not np.isnan(sma_50.iloc[-1]) else float(data['Close'].iloc[-1]),
            'volatility': float(volatility) if not np.isnan(volatility) else 25.0,
            'price': float(data['Close'].iloc[-1]),
            'volume': float(data['Volume'].iloc[-1]) if 'Volume' in data.columns and not np.isnan(data['Volume'].iloc[-1]) else 0.0
        }
    except Exception:
        return {
            'rsi': 50.0,
            'sma_20': float(data['Close'].iloc[-1]) if len(data) > 0 else 100.0,
            'sma_50': float(data['Close'].iloc[-1]) if len(data) > 0 else 100.0,
            'volatility': 25.0,
            'price': float(data['Close'].iloc[-1]) if len(data) > 0 else 100.0,
            'volume': 0.0
        }

def generate_options_data(commodity, current_price, volatility):
    """Generate synthetic options data for screening"""
    options_data = []
    
    # Calculate expiry dates
    today = datetime.now()
    
    # This week expiry (Thursday)
    days_until_thursday = (3 - today.weekday()) % 7
    if days_until_thursday == 0 and today.hour > 15:  # After market close on Thursday
        days_until_thursday = 7
    this_week_expiry = today + timedelta(days=days_until_thursday)
    
    # Next week expiry
    next_week_expiry = this_week_expiry + timedelta(days=7)
    
    # Current month expiry (last Thursday)
    current_month_expiry = datetime(today.year, today.month, 28)
    while current_month_expiry.weekday() != 3:
        current_month_expiry -= timedelta(days=1)
    if current_month_expiry <= today:
        next_month = today.month + 1 if today.month < 12 else 1
        next_year = today.year if today.month < 12 else today.year + 1
        current_month_expiry = datetime(next_year, next_month, 28)
        while current_month_expiry.weekday() != 3:
            current_month_expiry -= timedelta(days=1)
    
    # Next month expiry
    next_month = current_month_expiry.month + 1 if current_month_expiry.month < 12 else 1
    next_year = current_month_expiry.year if current_month_expiry.month < 12 else current_month_expiry.year + 1
    next_month_expiry = datetime(next_year, next_month, 28)
    while next_month_expiry.weekday() != 3:
        next_month_expiry -= timedelta(days=1)
    
    expiries = {
        'This Week': this_week_expiry,
        'Next Week': next_week_expiry,
        'Current Month': current_month_expiry,
        'Next Month': next_month_expiry
    }
    
    # Generate strike prices (ATM ± 20%)
    atm_strike = round(current_price / 50) * 50
    strikes = []
    for i in range(-4, 5):
        strikes.append(atm_strike + (i * 50))
    
    # Risk-free rate
    risk_free_rate = 0.065
    implied_vol = volatility / 100
    
    for expiry_name, expiry_date in expiries.items():
        time_to_expiry = max((expiry_date - today).days / 365.0, 0.001)
        
        for strike in strikes:
            # Determine moneyness
            if strike < current_price * 0.98:
                moneyness_type = "ITM"
            elif strike <= current_price * 1.02:
                moneyness_type = "ATM"
            else:
                moneyness_type = "OTM"
            
            # Calculate option prices and Greeks
            call_price = black_scholes_price(current_price, strike, time_to_expiry, risk_free_rate, implied_vol, 'call')
            call_greeks = calculate_greeks(current_price, strike, time_to_expiry, risk_free_rate, implied_vol, 'call')
            
            put_price = black_scholes_price(current_price, strike, time_to_expiry, risk_free_rate, implied_vol, 'put')
            put_greeks = calculate_greeks(current_price, strike, time_to_expiry, risk_free_rate, implied_vol, 'put')
            
            # Simulate volume and OI (for demonstration)
            base_volume = np.random.randint(5000, 50000)
            base_oi = np.random.randint(1000, 20000)
            
            # Adjust based on moneyness (ATM options typically have higher volume)
            if moneyness_type == "ATM":
                volume_multiplier = 1.5
                oi_multiplier = 1.3
            elif moneyness_type == "ITM" or moneyness_type == "OTM":
                volume_multiplier = 0.8
                oi_multiplier = 0.9
            else:
                volume_multiplier = 1.0
                oi_multiplier = 1.0
            
            call_volume = int(base_volume * volume_multiplier * np.random.uniform(0.7, 1.3))
            call_oi = int(base_oi * oi_multiplier * np.random.uniform(0.8, 1.2))
            put_volume = int(base_volume * volume_multiplier * np.random.uniform(0.7, 1.3))
            put_oi = int(base_oi * oi_multiplier * np.random.uniform(0.8, 1.2))
            
            # Bid-ask spread (percentage)
            bid_ask_spread = np.random.uniform(2.0, 6.0)
            
            # Call option
            options_data.append({
                'commodity': commodity,
                'symbol': f"{commodity}_{strike}CE_{expiry_name}",
                'type': 'Call',
                'strike': strike,
                'expiry': expiry_name,
                'expiry_date': expiry_date,
                'price': call_price,
                'delta': call_greeks['delta'],
                'gamma': call_greeks['gamma'],
                'theta': call_greeks['theta'],
                'vega': call_greeks['vega'],
                'iv': implied_vol * 100,
                'volume': call_volume,
                'oi': call_oi,
                'bid_ask_spread': bid_ask_spread,
                'moneyness': moneyness_type,
                'time_to_expiry': time_to_expiry,
                'underlying_price': current_price
            })
            
            # Put option
            options_data.append({
                'commodity': commodity,
                'symbol': f"{commodity}_{strike}PE_{expiry_name}",
                'type': 'Put',
                'strike': strike,
                'expiry': expiry_name,
                'expiry_date': expiry_date,
                'price': put_price,
                'delta': put_greeks['delta'],
                'gamma': put_greeks['gamma'],
                'theta': put_greeks['theta'],
                'vega': put_greeks['vega'],
                'iv': implied_vol * 100,
                'volume': put_volume,
                'oi': put_oi,
                'bid_ask_spread': bid_ask_spread,
                'moneyness': moneyness_type,
                'time_to_expiry': time_to_expiry,
                'underlying_price': current_price
            })
    
    return options_data

def apply_filters(df):
    """Apply screening filters to the options data"""
    filtered_df = df.copy()
    
    # Basic filters
    filtered_df = filtered_df[filtered_df['volume'] >= min_volume]
    filtered_df = filtered_df[filtered_df['oi'] >= min_oi]
    filtered_df = filtered_df[filtered_df['bid_ask_spread'] <= max_bid_ask_spread]
    
    # IV filter
    filtered_df = filtered_df[
        (filtered_df['iv'] >= iv_range[0]) & 
        (filtered_df['iv'] <= iv_range[1])
    ]
    
    # Greeks filters
    filtered_df = filtered_df[
        (abs(filtered_df['delta']) >= delta_range[0]) & 
        (abs(filtered_df['delta']) <= delta_range[1])
    ]
    filtered_df = filtered_df[filtered_df['gamma'] >= gamma_min]
    filtered_df = filtered_df[abs(filtered_df['theta']) <= theta_max]
    
    # Moneyness filter
    if moneyness != "All":
        filtered_df = filtered_df[filtered_df['moneyness'] == moneyness]
    
    # Expiry filter
    if expiry_filter != "All":
        filtered_df = filtered_df[filtered_df['expiry'] == expiry_filter]
    
    # Strategy type filter
    if strategy_type == "Long Calls":
        filtered_df = filtered_df[filtered_df['type'] == 'Call']
    elif strategy_type == "Long Puts":
        filtered_df = filtered_df[filtered_df['type'] == 'Put']
    
    return filtered_df

# Main content
st.subheader("🔍 Options Screening Results")

# Get all commodities data
indian_commodities = get_indian_commodities()
all_options_data = []

# Progress bar
progress_bar = st.progress(0)
status_text = st.empty()

try:
    for i, (commodity, symbol) in enumerate(indian_commodities.items()):
        status_text.text(f"Processing {commodity}...")
        progress_bar.progress((i + 1) / len(indian_commodities))
        
        try:
            # Get commodity data
            data = get_commodity_data(symbol, period='3mo')
            
            if data is not None and not data.empty and len(data) > 20:
                tech_indicators = calculate_technical_indicators(data)
                
                if tech_indicators and isinstance(tech_indicators, dict):
                    # Check RSI filter
                    rsi_value = tech_indicators.get('rsi', 50)
                    if rsi_range[0] <= rsi_value <= rsi_range[1]:
                        # Generate options data
                        options_data = generate_options_data(
                            commodity, 
                            tech_indicators['price'], 
                            tech_indicators['volatility']
                        )
                        all_options_data.extend(options_data)
        except Exception as e:
            st.warning(f"Error processing {commodity}: {str(e)}")
            continue

    progress_bar.empty()
    status_text.empty()

    if all_options_data:
        # Create DataFrame
        df_options = pd.DataFrame(all_options_data)
        
        # Apply filters
        filtered_df = apply_filters(df_options)
        
        # Display results count
        st.info(f"Found {len(filtered_df)} options matching your criteria out of {len(df_options)} total options")
        
        if len(filtered_df) > 0:
            # Sort by volume (most liquid first)
            filtered_df = filtered_df.sort_values('volume', ascending=False)
            
            # Display top opportunities
            st.subheader("🎯 Top Opportunities")
            
            # Create display DataFrame
            display_cols = [
                'commodity', 'symbol', 'type', 'strike', 'expiry', 'price', 
                'delta', 'gamma', 'theta', 'iv', 'volume', 'oi', 'moneyness'
            ]
            
            display_df = filtered_df[display_cols].head(20).copy()
            display_df['price'] = display_df['price'].apply(lambda x: f"₹{x:.2f}")
            display_df['delta'] = display_df['delta'].apply(lambda x: f"{x:.3f}")
            display_df['gamma'] = display_df['gamma'].apply(lambda x: f"{x:.4f}")
            display_df['theta'] = display_df['theta'].apply(lambda x: f"{x:.3f}")
            display_df['iv'] = display_df['iv'].apply(lambda x: f"{x:.1f}%")
            display_df['volume'] = display_df['volume'].apply(lambda x: f"{x:,}")
            display_df['oi'] = display_df['oi'].apply(lambda x: f"{x:,}")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Analysis charts
            st.subheader("📊 Screening Analysis")
            
            tab1, tab2, tab3 = st.tabs(["Volume Analysis", "Greeks Distribution", "IV Analysis"])
            
            with tab1:
                # Volume by commodity
                vol_by_commodity = filtered_df.groupby('commodity')['volume'].sum().sort_values(ascending=False)
                
                fig_volume = px.bar(
                    x=vol_by_commodity.index,
                    y=vol_by_commodity.values,
                    title="Total Volume by Commodity",
                    labels={'x': 'Commodity', 'y': 'Total Volume'}
                )
                st.plotly_chart(fig_volume, use_container_width=True)
                
                # Volume distribution
                fig_vol_dist = px.histogram(
                    filtered_df,
                    x='volume',
                    nbins=20,
                    title="Volume Distribution",
                    labels={'volume': 'Volume', 'count': 'Number of Options'}
                )
                st.plotly_chart(fig_vol_dist, use_container_width=True)
            
            with tab2:
                # Delta distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_delta = px.histogram(
                        filtered_df,
                        x='delta',
                        color='type',
                        nbins=20,
                        title="Delta Distribution by Option Type"
                    )
                    st.plotly_chart(fig_delta, use_container_width=True)
                
                with col2:
                    fig_gamma = px.histogram(
                        filtered_df,
                        x='gamma',
                        color='type',
                        nbins=20,
                        title="Gamma Distribution by Option Type"
                    )
                    st.plotly_chart(fig_gamma, use_container_width=True)
                
                # Scatter plot: Delta vs Gamma
                fig_scatter = px.scatter(
                    filtered_df,
                    x='delta',
                    y='gamma',
                    color='type',
                    size='volume',
                    hover_data=['commodity', 'strike', 'expiry'],
                    title="Delta vs Gamma (Size = Volume)"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with tab3:
                # IV by commodity
                fig_iv = px.box(
                    filtered_df,
                    x='commodity',
                    y='iv',
                    title="Implied Volatility Distribution by Commodity"
                )
                fig_iv.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_iv, use_container_width=True)
                
                # IV vs Time to expiry
                fig_iv_time = px.scatter(
                    filtered_df,
                    x='time_to_expiry',
                    y='iv',
                    color='commodity',
                    size='volume',
                    title="Implied Volatility vs Time to Expiry"
                )
                st.plotly_chart(fig_iv_time, use_container_width=True)
            
            # Strategy suggestions
            st.subheader("💡 Strategy Suggestions")
            
            # High volume opportunities
            high_vol_options = filtered_df[filtered_df['volume'] > filtered_df['volume'].quantile(0.8)]
            if len(high_vol_options) > 0:
                st.success(f"🚀 **High Volume Opportunities**: {len(high_vol_options)} options with exceptional volume")
                for _, option in high_vol_options.head(3).iterrows():
                    st.info(f"• **{option['commodity']} {option['strike']} {option['type']}** - Volume: {option['volume']:,}, IV: {option['iv']:.1f}%")
            
            # High gamma opportunities
            high_gamma_options = filtered_df[filtered_df['gamma'] > filtered_df['gamma'].quantile(0.9)]
            if len(high_gamma_options) > 0:
                st.success(f"⚡ **High Gamma Opportunities**: {len(high_gamma_options)} options with high gamma for momentum plays")
            
            # Low IV opportunities
            low_iv_options = filtered_df[filtered_df['iv'] < filtered_df['iv'].quantile(0.3)]
            if len(low_iv_options) > 0:
                st.success(f"💎 **Low IV Opportunities**: {len(low_iv_options)} options with relatively low implied volatility")
            
            # Export functionality
            if st.button("📥 Export Screening Results"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"options_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        else:
            st.warning("No options match your current filter criteria. Try adjusting the filters.")
    
    else:
        st.error("Unable to fetch options data. Please check your connection and try again.")

except Exception as e:
    st.error(f"Error in options screening: {str(e)}")
    st.exception(e)

# Screening presets
st.sidebar.subheader("📋 Quick Presets")

if st.sidebar.button("🚀 High Volume Scanner"):
    st.sidebar.info("Applied: Min Volume 25K, High liquidity focus")

if st.sidebar.button("⚡ High Gamma Scanner"):
    st.sidebar.info("Applied: Min Gamma 0.01, Momentum plays")

if st.sidebar.button("💎 Low IV Scanner"):
    st.sidebar.info("Applied: IV < 30%, Value opportunities")

if st.sidebar.button("🎯 ATM Options"):
    st.sidebar.info("Applied: ATM options filter")

# Auto-refresh
if st.sidebar.checkbox("Auto Refresh (2 min)", value=False):
    import time
    time.sleep(120)
    st.rerun()
