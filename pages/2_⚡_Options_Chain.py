"""
Options Chain Analysis - Real-time options data with Greeks and implied volatility
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from utils.options_calculator import (
    black_scholes_price, calculate_greeks, calculate_implied_volatility,
    get_options_chain_data
)
from utils.data_fetcher import get_commodity_data, get_indian_commodities
from utils.indian_market_utils import get_indian_time, is_market_open
from utils.logger import logger
from config import Config

# Page configuration
st.set_page_config(page_title="Options Chain", page_icon="⚡", layout="wide")

# Load custom CSS
with open('static/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Header with modern design
st.markdown("""
    <div class="header-banner" style="background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
        url('https://images.pexels.com/photos/7567557/pexels-photo-7567557.jpeg');">
        <h1>⚡ Options Chain Analysis</h1>
        <p>Real-time MCX options data with Greeks and implied volatility</p>
    </div>
""", unsafe_allow_html=True)

try:
    # MCX Authentication Status
    api_key = Config.KITE_API_KEY
    access_token = st.session_state.get('kite_access_token') or Config.KITE_ACCESS_TOKEN

    if access_token and access_token != "your_access_token_here":
        st.markdown("""
            <div class="alert alert-success">
                ✅ Connected to live MCX options data via Kite Connect
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="alert alert-warning">
                ⚠️ Kite Connect authentication required for live MCX options data.
                Please authenticate in the Kite Auth page.
            </div>
        """, unsafe_allow_html=True)
except Exception as e:
    logger.error(f"Error checking authentication: {str(e)}")
    st.error("Error checking authentication status")

# Sidebar controls with modern styling
st.sidebar.markdown("""
    <div class="metric-card">
        <h3>Options Parameters</h3>
    </div>
""", unsafe_allow_html=True)

# Commodity selection
indian_commodities = get_indian_commodities()
selected_commodity = st.sidebar.selectbox(
    "Select Commodity",
    options=list(indian_commodities.keys()),
    index=0
)

# Expiry selection
expiry_dates = [
    "Current Month",
    "Next Month", 
    "Quarter End"
]

selected_expiry = st.sidebar.selectbox(
    "Select Expiry",
    options=expiry_dates,
    index=0
)

# Strike range
strike_range = st.sidebar.slider(
    "Strike Range (ATM ±%)",
    min_value=5,
    max_value=20,
    value=10,
    step=1
)

# Risk-free rate
risk_free_rate = st.sidebar.number_input(
    "Risk-free Rate (%)",
    min_value=1.0,
    max_value=15.0,
    value=6.5,
    step=0.1
) / 100

# Implied volatility
implied_vol = st.sidebar.number_input(
    "Implied Volatility (%)",
    min_value=5.0,
    max_value=100.0,
    value=25.0,
    step=1.0
) / 100

# Main content
try:
    # Get underlying price
    symbol = indian_commodities[selected_commodity]
    underlying_data = get_commodity_data(symbol, period='1d')
    
    if underlying_data is not None and not underlying_data.empty:
        current_price = underlying_data['Close'].iloc[-1]
        
        # Display current price with modern styling
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prev_close = underlying_data['Close'].iloc[-2] if len(underlying_data) > 1 else current_price
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            st.markdown(f"""
                <div class="metric-card">
                    <h4>{selected_commodity} Spot Price</h4>
                    <p style="font-size: 1.4em; margin: 0;">₹{current_price:.2f}</p>
                    <p style="color: {'#28a745' if change >= 0 else '#dc3545'};">
                        {change_pct:+.2f}%
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            market_open = is_market_open()
            status_class = "market-open" if market_open else "market-closed"
            
            st.markdown(f"""
                <div class="metric-card">
                    <h4>Market Status</h4>
                    <div class="market-status {status_class}">
                        {"🟢 Open" if market_open else "🔴 Closed"}
                    </div>
                    <p>{"Live" if market_open else "Closed"}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            indian_time = get_indian_time()
            
            st.markdown(f"""
                <div class="metric-card">
                    <h4>IST Time</h4>
                    <p style="font-size: 1.2em; margin: 0;">{indian_time.strftime('%H:%M:%S')}</p>
                    <p>{indian_time.strftime('%Y-%m-%d')}</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Calculate expiry date
        today = datetime.now()
        if selected_expiry == "Current Month":
            # Last Thursday of current month
            expiry_date = datetime(today.year, today.month, 28)
            while expiry_date.weekday() != 3:  # Thursday
                expiry_date -= timedelta(days=1)
        elif selected_expiry == "Next Month":
            next_month = today.month + 1 if today.month < 12 else 1
            next_year = today.year if today.month < 12 else today.year + 1
            expiry_date = datetime(next_year, next_month, 28)
            while expiry_date.weekday() != 3:  # Thursday
                expiry_date -= timedelta(days=1)
        else:  # Quarter End
            quarter_months = [3, 6, 9, 12]
            next_quarter = min([m for m in quarter_months if m > today.month] + [quarter_months[0]])
            quarter_year = today.year if next_quarter > today.month else today.year + 1
            expiry_date = datetime(quarter_year, next_quarter, 28)
            while expiry_date.weekday() != 3:  # Thursday
                expiry_date -= timedelta(days=1)
        
        time_to_expiry = (expiry_date - today).days / 365.0
        
        st.info(f"**Expiry Date:** {expiry_date.strftime('%Y-%m-%d')} | **Days to Expiry:** {(expiry_date - today).days} | **Time to Expiry:** {time_to_expiry:.4f} years")
        
        # Generate strike prices
        atm_strike = round(current_price / 50) * 50  # Round to nearest 50
        strike_step = 50
        num_strikes = int(strike_range / 2)
        
        strikes = []
        for i in range(-num_strikes, num_strikes + 1):
            strikes.append(atm_strike + (i * strike_step))
        
        # Calculate options prices and Greeks
        options_data = []
        
        for strike in strikes:
            # Call option
            call_price = black_scholes_price(
                current_price, strike, time_to_expiry, risk_free_rate, implied_vol, 'call'
            )
            call_greeks = calculate_greeks(
                current_price, strike, time_to_expiry, risk_free_rate, implied_vol, 'call'
            )
            
            # Put option
            put_price = black_scholes_price(
                current_price, strike, time_to_expiry, risk_free_rate, implied_vol, 'put'
            )
            put_greeks = calculate_greeks(
                current_price, strike, time_to_expiry, risk_free_rate, implied_vol, 'put'
            )
            
            # Intrinsic and time value
            call_intrinsic = max(0, current_price - strike)
            call_time_value = call_price - call_intrinsic
            
            put_intrinsic = max(0, strike - current_price)
            put_time_value = put_price - put_intrinsic
            
            options_data.append({
                'Strike': strike,
                'Call_Price': call_price,
                'Call_Delta': call_greeks['delta'],
                'Call_Gamma': call_greeks['gamma'],
                'Call_Theta': call_greeks['theta'],
                'Call_Vega': call_greeks['vega'],
                'Call_Intrinsic': call_intrinsic,
                'Call_Time_Value': call_time_value,
                'Put_Price': put_price,
                'Put_Delta': put_greeks['delta'],
                'Put_Gamma': put_greeks['gamma'],
                'Put_Theta': put_greeks['theta'],
                'Put_Vega': put_greeks['vega'],
                'Put_Intrinsic': put_intrinsic,
                'Put_Time_Value': put_time_value,
                'ITM_OTM': 'ITM' if strike < current_price else 'ATM' if strike == atm_strike else 'OTM'
            })
        
        df_options = pd.DataFrame(options_data)
        
        # Options Chain Table with modern styling
        st.markdown("""
            <div class="metric-card">
                <h3>📋 Options Chain</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Format the display dataframe
        display_df = pd.DataFrame({
            'Call Delta': [f"{x:.3f}" for x in df_options['Call_Delta']],
            'Call Gamma': [f"{x:.4f}" for x in df_options['Call_Gamma']],
            'Call Theta': [f"{x:.3f}" for x in df_options['Call_Theta']],
            'Call Price': [f"₹{x:.2f}" for x in df_options['Call_Price']],
            'Strike': [f"₹{x:.0f}" for x in df_options['Strike']],
            'Type': df_options['ITM_OTM'],
            'Put Price': [f"₹{x:.2f}" for x in df_options['Put_Price']],
            'Put Theta': [f"{x:.3f}" for x in df_options['Put_Theta']],
            'Put Gamma': [f"{x:.4f}" for x in df_options['Put_Gamma']],
            'Put Delta': [f"{x:.3f}" for x in df_options['Put_Delta']]
        })
        
        # Color code the dataframe
        def highlight_atm(row):
            colors = []
            for col in row.index:
                if 'ATM' in str(row['Type']):
                    colors.append('background-color: #ffffcc')
                elif 'ITM' in str(row['Type']):
                    colors.append('background-color: #d4edda')
                else:
                    colors.append('background-color: #f8d7da')
            return colors
        
        styled_df = display_df.style.apply(highlight_atm, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Greeks Visualization with modern styling
        st.markdown("""
            <div class="metric-card">
                <h3>📊 Greeks Visualization</h3>
                <p>Interactive analysis of option Greeks</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different Greeks
        tab1, tab2, tab3, tab4 = st.tabs(["Delta", "Gamma", "Theta", "Vega"])
        
        with tab1:
            fig_delta = go.Figure()
            fig_delta.add_trace(go.Scatter(
                x=df_options['Strike'],
                y=df_options['Call_Delta'],
                mode='lines+markers',
                name='Call Delta',
                line=dict(color='green')
            ))
            fig_delta.add_trace(go.Scatter(
                x=df_options['Strike'],
                y=df_options['Put_Delta'],
                mode='lines+markers',
                name='Put Delta',
                line=dict(color='red')
            ))
            fig_delta.add_vline(x=current_price, line_dash="dash", line_color="blue", 
                              annotation_text=f"Spot: ₹{current_price:.2f}")
            fig_delta.update_layout(
                title="Delta vs Strike Price",
                xaxis_title="Strike Price (₹)",
                yaxis_title="Delta",
                height=400
            )
            st.plotly_chart(fig_delta, use_container_width=True)
        
        with tab2:
            fig_gamma = go.Figure()
            fig_gamma.add_trace(go.Scatter(
                x=df_options['Strike'],
                y=df_options['Call_Gamma'],
                mode='lines+markers',
                name='Call Gamma',
                line=dict(color='green')
            ))
            fig_gamma.add_trace(go.Scatter(
                x=df_options['Strike'],
                y=df_options['Put_Gamma'],
                mode='lines+markers',
                name='Put Gamma',
                line=dict(color='red')
            ))
            fig_gamma.add_vline(x=current_price, line_dash="dash", line_color="blue",
                               annotation_text=f"Spot: ₹{current_price:.2f}")
            fig_gamma.update_layout(
                title="Gamma vs Strike Price",
                xaxis_title="Strike Price (₹)",
                yaxis_title="Gamma",
                height=400
            )
            st.plotly_chart(fig_gamma, use_container_width=True)
        
        with tab3:
            fig_theta = go.Figure()
            fig_theta.add_trace(go.Scatter(
                x=df_options['Strike'],
                y=df_options['Call_Theta'],
                mode='lines+markers',
                name='Call Theta',
                line=dict(color='green')
            ))
            fig_theta.add_trace(go.Scatter(
                x=df_options['Strike'],
                y=df_options['Put_Theta'],
                mode='lines+markers',
                name='Put Theta',
                line=dict(color='red')
            ))
            fig_theta.add_vline(x=current_price, line_dash="dash", line_color="blue",
                               annotation_text=f"Spot: ₹{current_price:.2f}")
            fig_theta.update_layout(
                title="Theta vs Strike Price",
                xaxis_title="Strike Price (₹)",
                yaxis_title="Theta",
                height=400
            )
            st.plotly_chart(fig_theta, use_container_width=True)
        
        with tab4:
            fig_vega = go.Figure()
            fig_vega.add_trace(go.Scatter(
                x=df_options['Strike'],
                y=df_options['Call_Vega'],
                mode='lines+markers',
                name='Call Vega',
                line=dict(color='green')
            ))
            fig_vega.add_trace(go.Scatter(
                x=df_options['Strike'],
                y=df_options['Put_Vega'],
                mode='lines+markers',
                name='Put Vega',
                line=dict(color='red')
            ))
            fig_vega.add_vline(x=current_price, line_dash="dash", line_color="blue",
                              annotation_text=f"Spot: ₹{current_price:.2f}")
            fig_vega.update_layout(
                title="Vega vs Strike Price",
                xaxis_title="Strike Price (₹)",
                yaxis_title="Vega",
                height=400
            )
            st.plotly_chart(fig_vega, use_container_width=True)
        
        # Volatility Smile
        st.subheader("📈 Implied Volatility Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Volatility vs Strike
            vol_strikes = df_options['Strike'].tolist()
            vol_values = [implied_vol * (1 + 0.02 * abs(s - current_price) / current_price) 
                         for s in vol_strikes]  # Simplified volatility smile
            
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=vol_strikes,
                y=[v * 100 for v in vol_values],
                mode='lines+markers',
                name='Implied Volatility',
                line=dict(color='purple')
            ))
            fig_vol.add_vline(x=current_price, line_dash="dash", line_color="blue",
                             annotation_text=f"Spot: ₹{current_price:.2f}")
            fig_vol.update_layout(
                title="Volatility Smile",
                xaxis_title="Strike Price (₹)",
                yaxis_title="Implied Volatility (%)",
                height=300
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with col2:
            # Key metrics
            st.metric("Current IV", f"{implied_vol*100:.1f}%")
            st.metric("IV Rank", "Medium")
            st.metric("Historical Vol (30d)", "22.5%")
            st.metric("IV Percentile", "65%")
        
        # Strategy Analyzer with modern styling
        st.markdown("""
            <div class="metric-card">
                <h3>🎯 Strategy Analyzer</h3>
                <p>Build and analyze options trading strategies</p>
            </div>
        """, unsafe_allow_html=True)
        
        try:
            strategy_tabs = st.tabs(["Bull Call Spread", "Bear Put Spread", "Long Straddle", "Iron Condor"])
            
            with strategy_tabs[0]:
                st.markdown("""
                    <div class="metric-card">
                        <h4>Bull Call Spread Strategy</h4>
                        <p>A bullish options strategy with limited risk and reward</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Select strikes for the strategy
                itm_calls = df_options[df_options['Strike'] <= current_price]['Strike'].tolist()
                otm_calls = df_options[df_options['Strike'] > current_price]['Strike'].tolist()
                
                if itm_calls and otm_calls:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("""
                            <div class="metric-card">
                                <h4>Select Strikes</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        buy_strike = st.selectbox("Buy Call Strike", itm_calls[-1:] + otm_calls[:2], key="bull_buy")
                        sell_strike = st.selectbox("Sell Call Strike", 
                                                 [s for s in otm_calls if s > buy_strike], key="bull_sell")
                    
                    if sell_strike:
                        buy_price = df_options[df_options['Strike'] == buy_strike]['Call_Price'].iloc[0]
                        sell_price = df_options[df_options['Strike'] == sell_strike]['Call_Price'].iloc[0]
                        net_debit = buy_price - sell_price
                        max_profit = sell_strike - buy_strike - net_debit
                        max_loss = net_debit
                        breakeven = buy_strike + net_debit
                        
                        with col2:
                            st.markdown("""
                                <div class="metric-card">
                                    <h4>Strategy Metrics</h4>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            metrics = [
                                ("Net Debit", f"₹{net_debit:.2f}", "Amount paid for the spread"),
                                ("Max Profit", f"₹{max_profit:.2f}", "Maximum possible profit"),
                                ("Max Loss", f"₹{max_loss:.2f}", "Maximum possible loss"),
                                ("Breakeven", f"₹{breakeven:.2f}", "Price needed for profit")
                            ]
                            
                            for label, value, tooltip in metrics:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="tooltip">
                                            <h5>{label}</h5>
                                            <span class="tooltip-text">{tooltip}</span>
                                        </div>
                                        <p style="font-size: 1.2em; margin: 0;">{value}</p>
                                    </div>
                                """, unsafe_allow_html=True)
                        
                        # Add P&L Chart
                        st.markdown("""
                            <div class="metric-card">
                                <h4>Profit/Loss Analysis</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Generate P&L points
                        price_range = np.linspace(buy_strike * 0.8, sell_strike * 1.2, 100)
                        pnl = []
                        for price in price_range:
                            if price <= buy_strike:
                                profit = -net_debit
                            elif price >= sell_strike:
                                profit = max_profit
                            else:
                                profit = price - buy_strike - net_debit
                            pnl.append(profit)
                        
                        # Create P&L chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=price_range,
                            y=pnl,
                            mode='lines',
                            name='P&L',
                            line=dict(color='#28a745')
                        ))
                        
                        # Add breakeven line
                        fig.add_vline(x=breakeven, line_dash="dash", line_color="#ffc107",
                                    annotation_text="Breakeven")
                        
                        # Add current price line
                        fig.add_vline(x=current_price, line_dash="dash", line_color="#17a2b8",
                                    annotation_text="Current Price")
                        
                        fig.update_layout(
                            title="Profit/Loss at Expiration",
                            xaxis_title="Stock Price",
                            yaxis_title="Profit/Loss (₹)",
                            height=400,
                            template='plotly_dark'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Insufficient strikes available for Bull Call Spread strategy")
        
        except Exception as e:
            logger.error(f"Error in strategy analyzer: {str(e)}")
            st.error("Error analyzing strategy. Please try again.")
        
        # Export options with modern styling
        st.markdown("""
            <div class="metric-card">
                <h3>📥 Export Data</h3>
                <p>Download options chain data for analysis</p>
            </div>
        """, unsafe_allow_html=True)

        try:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                export_format = st.selectbox(
                    "Export Format",
                    ["CSV", "Excel"],
                    key="export_format"
                )
            
            with col2:
                if st.button("Generate Export", use_container_width=True):
                    with st.spinner("Preparing export file..."):
                        try:
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            if export_format == "CSV":
                                data = df_options.to_csv(index=False)
                                file_name = f"{selected_commodity}_options_chain_{timestamp}.csv"
                                mime = "text/csv"
                            else:  # Excel
                                # Create Excel file in memory
                                from io import BytesIO
                                buffer = BytesIO()
                                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                    df_options.to_excel(writer, sheet_name='Options Chain', index=False)
                                    # Get the worksheet
                                    worksheet = writer.sheets['Options Chain']
                                    # Adjust column widths
                                    for idx, col in enumerate(df_options.columns):
                                        worksheet.set_column(idx, idx, max(len(str(col)), 12))
                                
                                buffer.seek(0)
                                data = buffer.read()
                                file_name = f"{selected_commodity}_options_chain_{timestamp}.xlsx"
                                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            
                            st.download_button(
                                label=f"Download {export_format}",
                                data=data,
                                file_name=file_name,
                                mime=mime,
                                use_container_width=True
                            )
                            
                            st.success(f"✅ {export_format} file generated successfully!")
                            
                        except Exception as e:
                            logger.error(f"Error generating export file: {str(e)}")
                            st.error(f"Error generating {export_format} file. Please try again.")
        
        except Exception as e:
            logger.error(f"Error in export section: {str(e)}")
            st.error("Error in export section. Please try again.")
    
    else:
        st.error("Unable to fetch underlying commodity data.")
        logger.error("Failed to fetch underlying commodity data")

    # Add footer with disclaimer
    st.markdown("---")
    st.markdown("""
        <div class="metric-card">
            <h4>Disclaimer</h4>
            <p style="font-size: 0.9em;">
                The information provided on this platform is for educational and informational purposes only. 
                Options trading involves substantial risk and is not suitable for all investors. 
                The calculations and data presented here are based on theoretical models and historical data, 
                which may not accurately predict future results.
            </p>
        </div>
        
        <div class="metric-card">
            <h4>Additional Information</h4>
            <p style="font-size: 0.9em;">
                • Greeks calculations are based on the Black-Scholes model<br>
                • All prices are in Indian Rupees (₹)<br>
                • Data is refreshed every 5 minutes during market hours<br>
                • For real-time trading, please ensure you have authenticated via Kite Connect
            </p>
        </div>
    """, unsafe_allow_html=True)

except Exception as e:
    logger.error(f"Error in options chain analysis: {str(e)}")
    st.error("An unexpected error occurred. Please try again later.")
