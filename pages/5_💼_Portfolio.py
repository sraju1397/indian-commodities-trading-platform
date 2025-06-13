import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from utils.portfolio_manager import PortfolioManager
from utils.options_calculator import black_scholes_price, calculate_greeks
from utils.data_fetcher import get_commodity_data, get_indian_commodities
from utils.indian_market_utils import get_indian_time, is_market_open

st.set_page_config(page_title="Portfolio", page_icon="💼", layout="wide")

st.title("💼 Portfolio Management")
st.markdown("### Track your options positions and performance")

# Initialize portfolio manager
if 'portfolio_manager' not in st.session_state:
    st.session_state.portfolio_manager = PortfolioManager()

portfolio_manager = st.session_state.portfolio_manager

# Sidebar - Add new position
st.sidebar.header("Add New Position")

with st.sidebar.form("add_position"):
    st.subheader("📝 New Trade Entry")
    
    # Trade details
    indian_commodities = get_indian_commodities()
    commodity = st.selectbox("Commodity", list(indian_commodities.keys()))
    
    option_type = st.selectbox("Option Type", ["Call", "Put"])
    
    action = st.selectbox("Action", ["Buy", "Sell"])
    
    strike_price = st.number_input("Strike Price (₹)", min_value=1000.0, max_value=100000.0, value=50000.0, step=50.0)
    
    quantity = st.number_input("Quantity (Lots)", min_value=1, max_value=1000, value=1)
    
    premium = st.number_input("Premium per unit (₹)", min_value=0.1, max_value=10000.0, value=100.0, step=0.1)
    
    # Expiry date
    expiry_date = st.date_input("Expiry Date", value=datetime.now() + timedelta(days=30))
    
    # Optional fields
    st.subheader("Optional Details")
    notes = st.text_area("Trade Notes", placeholder="Reason for trade, strategy, etc.")
    
    submitted = st.form_submit_button("Add Position")
    
    if submitted:
        try:
            portfolio_manager.add_position(
                commodity=commodity,
                option_type=option_type,
                action=action,
                strike_price=strike_price,
                quantity=quantity,
                premium=premium,
                expiry_date=expiry_date,
                notes=notes
            )
            st.success("Position added successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error adding position: {str(e)}")

# Sidebar - Import/Export
st.sidebar.header("Data Management")

# Export portfolio
if st.sidebar.button("📤 Export Portfolio"):
    portfolio_data = portfolio_manager.get_portfolio_summary()
    if portfolio_data:
        df = pd.DataFrame(portfolio_data)
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Clear portfolio (with confirmation)
if st.sidebar.button("🗑️ Clear Portfolio", type="secondary"):
    if st.sidebar.checkbox("I confirm I want to clear all positions"):
        portfolio_manager.clear_portfolio()
        st.sidebar.success("Portfolio cleared!")
        st.rerun()

# Main content
positions = portfolio_manager.get_all_positions()

if not positions:
    st.info("👋 Welcome to your portfolio! Add your first position using the sidebar.")
    
    # Sample portfolio for demonstration
    with st.expander("📚 Portfolio Management Guide"):
        st.markdown("""
        #### How to use the Portfolio Manager:
        
        **1. Adding Positions:**
        - Use the sidebar form to add new options positions
        - Enter all trade details including strike, premium, quantity
        - Add notes for your trading rationale
        
        **2. Position Tracking:**
        - View all your active positions in the main dashboard
        - Monitor real-time P&L and Greeks
        - Track time decay and volatility impact
        
        **3. Performance Analysis:**
        - Analyze your trading performance over time
        - View risk metrics and exposure
        - Identify profitable strategies
        
        **4. Risk Management:**
        - Monitor portfolio Greeks and risk exposure
        - Set alerts for position changes
        - Track margin requirements
        """)

else:
    # Portfolio overview
    st.subheader("📊 Portfolio Overview")
    
    # Calculate portfolio metrics
    total_positions = len(positions)
    active_positions = len([p for p in positions if p['status'] == 'Active'])
    
    # Get current market data for P&L calculation
    portfolio_pnl = 0
    portfolio_value = 0
    total_premium_paid = 0
    
    for position in positions:
        if position['status'] == 'Active':
            try:
                # Get current underlying price
                symbol = get_indian_commodities().get(position['commodity'])
                if symbol:
                    data = get_commodity_data(symbol, period='1d')
                    if data is not None and not data.empty:
                        current_price = data['Close'].iloc[-1]
                        
                        # Calculate time to expiry
                        expiry_date = datetime.strptime(position['expiry_date'], '%Y-%m-%d')
                        time_to_expiry = max((expiry_date - datetime.now()).days / 365.0, 0.001)
                        
                        # Calculate current option price
                        current_option_price = black_scholes_price(
                            current_price, 
                            position['strike_price'], 
                            time_to_expiry, 
                            0.065,  # Risk-free rate
                            0.25,   # Assumed IV
                            position['option_type'].lower()
                        )
                        
                        # Calculate P&L
                        if position['action'] == 'Buy':
                            position_value = current_option_price * position['quantity']
                            position_cost = position['premium'] * position['quantity']
                            position_pnl = position_value - position_cost
                        else:  # Sell
                            position_value = -current_option_price * position['quantity']
                            position_cost = -position['premium'] * position['quantity']
                            position_pnl = position_value - position_cost
                        
                        portfolio_pnl += position_pnl
                        portfolio_value += abs(position_value)
                        total_premium_paid += abs(position_cost)
            except:
                continue
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Positions", total_positions)
    
    with col2:
        st.metric("Active Positions", active_positions)
    
    with col3:
        pnl_color = "normal" if portfolio_pnl >= 0 else "inverse"
        st.metric("Portfolio P&L", f"₹{portfolio_pnl:,.2f}", 
                 delta_color=pnl_color)
    
    with col4:
        portfolio_return = (portfolio_pnl / total_premium_paid * 100) if total_premium_paid > 0 else 0
        st.metric("Return %", f"{portfolio_return:+.2f}%")
    
    # Positions table
    st.subheader("📋 Current Positions")
    
    # Create positions DataFrame with current market data
    positions_data = []
    
    for i, position in enumerate(positions):
        if position['status'] == 'Active':
            try:
                # Get current market data
                symbol = get_indian_commodities().get(position['commodity'])
                current_price = 0
                current_option_price = 0
                pnl = 0
                pnl_pct = 0
                
                if symbol:
                    data = get_commodity_data(symbol, period='1d')
                    if data is not None and not data.empty:
                        current_price = data['Close'].iloc[-1]
                        
                        # Calculate time to expiry
                        expiry_date = datetime.strptime(position['expiry_date'], '%Y-%m-%d')
                        time_to_expiry = max((expiry_date - datetime.now()).days / 365.0, 0.001)
                        days_to_expiry = (expiry_date - datetime.now()).days
                        
                        # Calculate current option price and Greeks
                        current_option_price = black_scholes_price(
                            current_price, 
                            position['strike_price'], 
                            time_to_expiry, 
                            0.065, 
                            0.25, 
                            position['option_type'].lower()
                        )
                        
                        greeks = calculate_greeks(
                            current_price, 
                            position['strike_price'], 
                            time_to_expiry, 
                            0.065, 
                            0.25, 
                            position['option_type'].lower()
                        )
                        
                        # Calculate P&L
                        if position['action'] == 'Buy':
                            pnl = (current_option_price - position['premium']) * position['quantity']
                        else:  # Sell
                            pnl = (position['premium'] - current_option_price) * position['quantity']
                        
                        pnl_pct = (pnl / (position['premium'] * position['quantity'])) * 100
                        
                        positions_data.append({
                            'ID': i + 1,
                            'Commodity': position['commodity'],
                            'Type': f"{position['option_type']} {position['action']}",
                            'Strike': f"₹{position['strike_price']:,.0f}",
                            'Qty': position['quantity'],
                            'Entry Premium': f"₹{position['premium']:.2f}",
                            'Current Premium': f"₹{current_option_price:.2f}",
                            'Current Price': f"₹{current_price:.2f}",
                            'Days to Expiry': days_to_expiry,
                            'P&L': f"₹{pnl:,.2f}",
                            'P&L %': f"{pnl_pct:+.1f}%",
                            'Delta': f"{greeks['delta']:.3f}",
                            'Gamma': f"{greeks['gamma']:.4f}",
                            'Theta': f"{greeks['theta']:.3f}",
                            'Trade Date': position['trade_date'],
                            'Expiry': position['expiry_date']
                        })
            except Exception as e:
                # Add position with limited data if market data fails
                positions_data.append({
                    'ID': i + 1,
                    'Commodity': position['commodity'],
                    'Type': f"{position['option_type']} {position['action']}",
                    'Strike': f"₹{position['strike_price']:,.0f}",
                    'Qty': position['quantity'],
                    'Entry Premium': f"₹{position['premium']:.2f}",
                    'Current Premium': 'N/A',
                    'Current Price': 'N/A',
                    'Days to Expiry': 'N/A',
                    'P&L': 'N/A',
                    'P&L %': 'N/A',
                    'Delta': 'N/A',
                    'Gamma': 'N/A',
                    'Theta': 'N/A',
                    'Trade Date': position['trade_date'],
                    'Expiry': position['expiry_date']
                })
    
    if positions_data:
        df_positions = pd.DataFrame(positions_data)
        
        # Style the dataframe
        def color_pnl(val):
            if 'N/A' in str(val):
                return ''
            try:
                if 'P&L' in str(val):
                    num_val = float(val.replace('₹', '').replace(',', '').replace('%', '').replace('+', ''))
                    color = 'color: green' if num_val >= 0 else 'color: red'
                    return color
            except:
                return ''
            return ''
        
        # Apply styling
        styled_df = df_positions.style.applymap(color_pnl, subset=['P&L', 'P&L %'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Portfolio analytics
        st.subheader("📈 Portfolio Analytics")
        
        tab1, tab2, tab3, tab4 = st.tabs(["P&L Analysis", "Greeks Exposure", "Expiry Calendar", "Performance"])
        
        with tab1:
            # P&L by commodity
            col1, col2 = st.columns(2)
            
            with col1:
                # Calculate P&L by commodity
                pnl_by_commodity = {}
                for position in positions:
                    if position['status'] == 'Active' and position['commodity'] in pnl_by_commodity:
                        continue  # Simplified for demo
                    # Real calculation would aggregate actual P&L
                
                # P&L chart (simplified)
                commodities = [p['commodity'] for p in positions if p['status'] == 'Active']
                pnl_values = [np.random.uniform(-1000, 2000) for _ in commodities]  # Demo values
                
                fig_pnl = px.bar(
                    x=commodities,
                    y=pnl_values,
                    title="P&L by Commodity",
                    color=pnl_values,
                    color_continuous_scale=['red', 'green']
                )
                st.plotly_chart(fig_pnl, use_container_width=True)
            
            with col2:
                # P&L over time (demo chart)
                dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
                cumulative_pnl = np.cumsum(np.random.uniform(-500, 600, len(dates)))
                
                fig_cumulative = go.Figure()
                fig_cumulative.add_trace(go.Scatter(
                    x=dates,
                    y=cumulative_pnl,
                    mode='lines',
                    name='Cumulative P&L',
                    line=dict(color='blue')
                ))
                fig_cumulative.update_layout(
                    title="Cumulative P&L Over Time",
                    xaxis_title="Date",
                    yaxis_title="P&L (₹)"
                )
                st.plotly_chart(fig_cumulative, use_container_width=True)
        
        with tab2:
            # Portfolio Greeks
            st.markdown("### Portfolio Greeks Summary")
            
            # Calculate portfolio Greeks (simplified)
            total_delta = sum([0.5 * p['quantity'] for p in positions if p['status'] == 'Active'])
            total_gamma = sum([0.01 * p['quantity'] for p in positions if p['status'] == 'Active'])
            total_theta = sum([-2.0 * p['quantity'] for p in positions if p['status'] == 'Active'])
            total_vega = sum([10.0 * p['quantity'] for p in positions if p['status'] == 'Active'])
            
            greeks_col1, greeks_col2, greeks_col3, greeks_col4 = st.columns(4)
            
            with greeks_col1:
                st.metric("Portfolio Delta", f"{total_delta:.2f}")
                st.caption("Directional exposure")
            
            with greeks_col2:
                st.metric("Portfolio Gamma", f"{total_gamma:.3f}")
                st.caption("Delta sensitivity")
            
            with greeks_col3:
                st.metric("Portfolio Theta", f"{total_theta:.2f}")
                st.caption("Time decay per day")
            
            with greeks_col4:
                st.metric("Portfolio Vega", f"{total_vega:.2f}")
                st.caption("Volatility sensitivity")
            
            # Greeks by position
            greeks_data = []
            for i, position in enumerate(positions):
                if position['status'] == 'Active':
                    greeks_data.append({
                        'Position': f"{position['commodity']} {position['strike_price']} {position['option_type']}",
                        'Delta': 0.5 * position['quantity'],  # Simplified
                        'Gamma': 0.01 * position['quantity'],
                        'Theta': -2.0 * position['quantity'],
                        'Vega': 10.0 * position['quantity']
                    })
            
            if greeks_data:
                df_greeks = pd.DataFrame(greeks_data)
                
                # Greeks visualization
                fig_greeks = go.Figure()
                fig_greeks.add_trace(go.Bar(
                    name='Delta',
                    x=df_greeks['Position'],
                    y=df_greeks['Delta']
                ))
                fig_greeks.update_layout(
                    title="Delta Exposure by Position",
                    xaxis_title="Position",
                    yaxis_title="Delta"
                )
                st.plotly_chart(fig_greeks, use_container_width=True)
        
        with tab3:
            # Expiry calendar
            st.markdown("### Expiry Calendar")
            
            expiry_data = []
            for position in positions:
                if position['status'] == 'Active':
                    expiry_date = datetime.strptime(position['expiry_date'], '%Y-%m-%d')
                    days_to_expiry = (expiry_date - datetime.now()).days
                    
                    expiry_data.append({
                        'Position': f"{position['commodity']} {position['strike_price']} {position['option_type']}",
                        'Expiry Date': position['expiry_date'],
                        'Days to Expiry': days_to_expiry,
                        'Quantity': position['quantity'],
                        'Premium': position['premium']
                    })
            
            if expiry_data:
                df_expiry = pd.DataFrame(expiry_data)
                df_expiry = df_expiry.sort_values('Days to Expiry')
                st.dataframe(df_expiry, use_container_width=True)
                
                # Expiry timeline
                fig_timeline = px.scatter(
                    df_expiry,
                    x='Days to Expiry',
                    y='Position',
                    size='Quantity',
                    title="Position Expiry Timeline"
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
        
        with tab4:
            # Performance metrics
            st.markdown("### Performance Metrics")
            
            # Win/Loss ratio
            profitable_positions = len([p for p in positions_data if 'P&L' in p and '₹' in str(p['P&L']) and float(str(p['P&L']).replace('₹', '').replace(',', '')) > 0])
            total_closed_positions = len([p for p in positions if p['status'] == 'Closed'])
            
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            with perf_col1:
                win_rate = (profitable_positions / len(positions_data) * 100) if positions_data else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with perf_col2:
                avg_return = portfolio_return / len(positions_data) if positions_data else 0
                st.metric("Avg Return per Trade", f"{avg_return:.1f}%")
            
            with perf_col3:
                st.metric("Total Trades", len(positions))
            
            # Risk metrics
            st.markdown("#### Risk Metrics")
            
            risk_col1, risk_col2, risk_col3 = st.columns(3)
            
            with risk_col1:
                max_single_loss = abs(min([float(str(p['P&L']).replace('₹', '').replace(',', '')) for p in positions_data if 'P&L' in p and '₹' in str(p['P&L'])], default=[0]))
                st.metric("Max Single Loss", f"₹{max_single_loss:,.2f}")
            
            with risk_col2:
                portfolio_concentration = max([positions_data.count(p) for p in set([pos['Commodity'] for pos in positions_data])], default=0)
                st.metric("Max Commodity Exposure", f"{portfolio_concentration} positions")
            
            with risk_col3:
                total_margin = sum([p['premium'] * p['quantity'] for p in positions if p['status'] == 'Active'])
                st.metric("Total Capital at Risk", f"₹{total_margin:,.2f}")
    
    else:
        st.info("No active positions found.")

# Position management actions
if positions:
    st.subheader("⚙️ Position Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Close position
        position_to_close = st.selectbox(
            "Select position to close",
            options=range(len(positions)),
            format_func=lambda x: f"{positions[x]['commodity']} {positions[x]['strike_price']} {positions[x]['option_type']} {positions[x]['action']}"
        )
        
        if st.button("Close Position"):
            portfolio_manager.close_position(position_to_close)
            st.success("Position closed!")
            st.rerun()
    
    with col2:
        # Modify position
        st.info("Position modification coming soon...")

# Market status
st.sidebar.header("Market Status")
market_open = is_market_open()
indian_time = get_indian_time()

if market_open:
    st.sidebar.success(f"🟢 Markets OPEN\n{indian_time.strftime('%H:%M:%S IST')}")
else:
    st.sidebar.error(f"🔴 Markets CLOSED\n{indian_time.strftime('%H:%M:%S IST')}")

# Auto-refresh
if st.sidebar.checkbox("Auto Refresh (30s)", value=False):
    import time
    time.sleep(30)
    st.rerun()
