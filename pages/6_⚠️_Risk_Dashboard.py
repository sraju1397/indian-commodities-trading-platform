import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from utils.portfolio_manager import PortfolioManager
from utils.risk_calculator import RiskCalculator
from utils.data_fetcher import get_commodity_data, get_indian_commodities
from utils.options_calculator import calculate_greeks, black_scholes_price
from utils.indian_market_utils import get_indian_time, is_market_open

st.set_page_config(page_title="Risk Dashboard", page_icon="⚠️", layout="wide")

st.title("⚠️ Risk Management Dashboard")
st.markdown("### Comprehensive risk analysis and portfolio monitoring")

# Initialize components
if 'portfolio_manager' not in st.session_state:
    st.session_state.portfolio_manager = PortfolioManager()

portfolio_manager = st.session_state.portfolio_manager
risk_calculator = RiskCalculator()

# Sidebar - Risk Parameters
st.sidebar.header("Risk Parameters")

# Risk tolerance settings
risk_tolerance = st.sidebar.selectbox(
    "Risk Tolerance Level",
    options=["Conservative", "Moderate", "Aggressive"],
    index=1
)

# Portfolio limits
max_portfolio_risk = st.sidebar.slider(
    "Max Portfolio Risk (%)",
    min_value=1.0,
    max_value=20.0,
    value=5.0,
    step=0.5
)

max_single_position = st.sidebar.slider(
    "Max Single Position (%)",
    min_value=1.0,
    max_value=15.0,
    value=3.0,
    step=0.5
)

# VaR parameters
var_confidence = st.sidebar.selectbox(
    "VaR Confidence Level",
    options=[90, 95, 99],
    index=1
)

var_horizon = st.sidebar.selectbox(
    "VaR Time Horizon (days)",
    options=[1, 5, 10, 20],
    index=0
)

# Stress test scenarios
st.sidebar.subheader("Stress Test Scenarios")

stress_scenarios = {
    "Market Crash": st.sidebar.checkbox("Market Crash (-20%)", value=True),
    "Volatility Spike": st.sidebar.checkbox("Volatility Spike (+50%)", value=True),
    "Interest Rate Change": st.sidebar.checkbox("Interest Rate Change (±2%)", value=False),
    "Commodity Shock": st.sidebar.checkbox("Commodity Shock (±30%)", value=True)
}

# Main content
positions = portfolio_manager.get_all_positions()
active_positions = [p for p in positions if p['status'] == 'Active']

if not active_positions:
    st.info("📊 No active positions found. Add positions in the Portfolio section to view risk analysis.")
    
    # Risk education content
    with st.expander("📚 Risk Management Fundamentals"):
        st.markdown("""
        #### Key Risk Metrics in Options Trading:
        
        **Value at Risk (VaR):**
        - Estimates potential loss over a specific time period at a given confidence level
        - Example: 1-day 95% VaR of ₹10,000 means 95% confidence that daily loss won't exceed ₹10,000
        
        **Greeks Risk:**
        - **Delta Risk:** Directional exposure to underlying price movements
        - **Gamma Risk:** Risk of delta changes, especially for short options
        - **Theta Risk:** Time decay impact on option values
        - **Vega Risk:** Sensitivity to volatility changes
        
        **Portfolio Risk:**
        - **Concentration Risk:** Over-exposure to single commodity or strategy
        - **Liquidity Risk:** Difficulty in closing positions during market stress
        - **Margin Risk:** Risk of margin calls due to adverse movements
        
        **Risk Management Best Practices:**
        - Diversify across commodities and strategies
        - Set position size limits (typically 2-5% per trade)
        - Use stop losses and profit targets
        - Monitor Greeks exposure regularly
        - Conduct regular stress testing
        """)

else:
    # Portfolio risk overview
    st.subheader("📊 Portfolio Risk Overview")
    
    # Calculate current portfolio metrics
    portfolio_value = 0
    total_margin = 0
    portfolio_pnl = 0
    
    # Get current market data for risk calculations
    risk_data = []
    
    for position in active_positions:
        try:
            # Get current underlying price
            symbol = get_indian_commodities().get(position['commodity'])
            if symbol:
                data = get_commodity_data(symbol, period='1mo')
                if data is not None and not data.empty:
                    current_price = data['Close'].iloc[-1]
                    
                    # Calculate time to expiry
                    expiry_date = datetime.strptime(position['expiry_date'], '%Y-%m-%d')
                    time_to_expiry = max((expiry_date - datetime.now()).days / 365.0, 0.001)
                    
                    # Calculate current option price and Greeks
                    current_option_price = black_scholes_price(
                        current_price, 
                        position['strike_price'], 
                        time_to_expiry, 
                        0.065,  # Risk-free rate
                        0.25,   # Assumed IV
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
                    
                    # Calculate position value and P&L
                    position_notional = current_option_price * position['quantity']
                    position_margin = position['premium'] * position['quantity']
                    
                    if position['action'] == 'Buy':
                        position_pnl = (current_option_price - position['premium']) * position['quantity']
                        portfolio_value += position_notional
                    else:  # Sell
                        position_pnl = (position['premium'] - current_option_price) * position['quantity']
                        portfolio_value += position_notional
                    
                    portfolio_pnl += position_pnl
                    total_margin += abs(position_margin)
                    
                    # Store risk data
                    risk_data.append({
                        'position': position,
                        'current_price': current_price,
                        'option_price': current_option_price,
                        'greeks': greeks,
                        'pnl': position_pnl,
                        'notional': position_notional,
                        'margin': position_margin,
                        'underlying_data': data
                    })
        except Exception as e:
            st.warning(f"Could not fetch risk data for {position['commodity']}: {str(e)}")
    
    if risk_data:
        # Display key risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            portfolio_return = (portfolio_pnl / total_margin * 100) if total_margin > 0 else 0
            st.metric("Portfolio Return", f"{portfolio_return:+.2f}%")
        
        with col2:
            portfolio_risk_pct = (abs(portfolio_pnl) / total_margin * 100) if total_margin > 0 else 0
            risk_status = "🟢 Low" if portfolio_risk_pct < 3 else "🟡 Medium" if portfolio_risk_pct < 7 else "🔴 High"
            st.metric("Current Risk Level", risk_status)
        
        with col3:
            total_delta = sum([rd['greeks']['delta'] * rd['position']['quantity'] for rd in risk_data])
            delta_status = "🟢 Neutral" if abs(total_delta) < 50 else "🟡 Moderate" if abs(total_delta) < 100 else "🔴 High"
            st.metric("Delta Exposure", f"{total_delta:.1f}", delta_status)
        
        with col4:
            total_theta = sum([rd['greeks']['theta'] * rd['position']['quantity'] for rd in risk_data])
            st.metric("Daily Theta Decay", f"₹{total_theta:.2f}")
        
        # Risk analysis tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 VaR Analysis", "📈 Greeks Risk", "🔥 Stress Testing", "⚡ Scenario Analysis", "📋 Risk Limits"])
        
        with tab1:
            st.subheader("Value at Risk (VaR) Analysis")
            
            # Calculate VaR using historical simulation
            var_results = risk_calculator.calculate_portfolio_var(
                risk_data, var_confidence, var_horizon
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### VaR Metrics")
                
                # Display VaR results
                for metric, value in var_results.items():
                    if isinstance(value, (int, float)):
                        if 'var' in metric.lower():
                            color = "🟢" if abs(value) < total_margin * 0.05 else "🟡" if abs(value) < total_margin * 0.1 else "🔴"
                            st.metric(f"{metric} {color}", f"₹{value:,.2f}")
                        else:
                            st.metric(metric, f"{value:.2f}%")
            
            with col2:
                # VaR distribution chart
                if 'var_distribution' in var_results:
                    fig_var = go.Figure()
                    
                    distribution = var_results['var_distribution']
                    fig_var.add_trace(go.Histogram(
                        x=distribution,
                        nbinsx=50,
                        name='P&L Distribution',
                        opacity=0.7
                    ))
                    
                    # Add VaR line
                    var_value = var_results.get('Portfolio VaR', 0)
                    fig_var.add_vline(
                        x=var_value,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"VaR: ₹{var_value:,.0f}"
                    )
                    
                    fig_var.update_layout(
                        title=f"{var_horizon}-Day {var_confidence}% VaR Distribution",
                        xaxis_title="P&L (₹)",
                        yaxis_title="Frequency",
                        height=400
                    )
                    st.plotly_chart(fig_var, use_container_width=True)
            
            # VaR by position
            st.subheader("VaR Contribution by Position")
            
            var_contributions = []
            for rd in risk_data:
                position = rd['position']
                # Calculate individual position VaR (simplified)
                position_var = risk_calculator.calculate_position_var(rd, var_confidence, var_horizon)
                
                var_contributions.append({
                    'Position': f"{position['commodity']} {position['strike_price']} {position['option_type']}",
                    'Individual VaR': f"₹{position_var:,.2f}",
                    'Portfolio Weight': f"{(abs(rd['margin']) / total_margin * 100):.1f}%",
                    'Risk Contribution': f"{(abs(position_var) / abs(var_results.get('Portfolio VaR', 1)) * 100):.1f}%"
                })
            
            if var_contributions:
                df_var = pd.DataFrame(var_contributions)
                st.dataframe(df_var, use_container_width=True)
        
        with tab2:
            st.subheader("Greeks Risk Analysis")
            
            # Calculate portfolio Greeks
            portfolio_greeks = risk_calculator.calculate_portfolio_greeks(risk_data)
            
            # Display portfolio Greeks
            greeks_col1, greeks_col2, greeks_col3, greeks_col4 = st.columns(4)
            
            with greeks_col1:
                delta = portfolio_greeks['total_delta']
                delta_risk = "🟢 Low" if abs(delta) < 50 else "🟡 Medium" if abs(delta) < 100 else "🔴 High"
                st.metric("Portfolio Delta", f"{delta:.2f}", delta_risk)
            
            with greeks_col2:
                gamma = portfolio_greeks['total_gamma']
                gamma_risk = "🟢 Low" if abs(gamma) < 0.5 else "🟡 Medium" if abs(gamma) < 1.0 else "🔴 High"
                st.metric("Portfolio Gamma", f"{gamma:.3f}", gamma_risk)
            
            with greeks_col3:
                theta = portfolio_greeks['total_theta']
                theta_risk = "🟢 Low" if abs(theta) < total_margin * 0.01 else "🟡 Medium" if abs(theta) < total_margin * 0.02 else "🔴 High"
                st.metric("Portfolio Theta", f"₹{theta:.2f}", theta_risk)
            
            with greeks_col4:
                vega = portfolio_greeks['total_vega']
                vega_risk = "🟢 Low" if abs(vega) < total_margin * 0.1 else "🟡 Medium" if abs(vega) < total_margin * 0.2 else "🔴 High"
                st.metric("Portfolio Vega", f"₹{vega:.2f}", vega_risk)
            
            # Greeks by position
            st.subheader("Greeks Breakdown by Position")
            
            greeks_data = []
            for rd in risk_data:
                position = rd['position']
                greeks = rd['greeks']
                
                greeks_data.append({
                    'Position': f"{position['commodity']} {position['strike_price']} {position['option_type']}",
                    'Quantity': position['quantity'],
                    'Delta': f"{greeks['delta'] * position['quantity']:.2f}",
                    'Gamma': f"{greeks['gamma'] * position['quantity']:.3f}",
                    'Theta': f"₹{greeks['theta'] * position['quantity']:.2f}",
                    'Vega': f"₹{greeks['vega'] * position['quantity']:.2f}"
                })
            
            if greeks_data:
                df_greeks = pd.DataFrame(greeks_data)
                st.dataframe(df_greeks, use_container_width=True)
                
                # Greeks visualization
                fig_greeks = go.Figure()
                
                # Add bars for each Greek
                fig_greeks.add_trace(go.Bar(
                    name='Delta',
                    x=[g['Position'] for g in greeks_data],
                    y=[float(g['Delta']) for g in greeks_data],
                    yaxis='y'
                ))
                
                fig_greeks.update_layout(
                    title="Delta Exposure by Position",
                    xaxis_title="Position",
                    yaxis_title="Delta",
                    height=400
                )
                fig_greeks.update_xaxis(tickangle=45)
                st.plotly_chart(fig_greeks, use_container_width=True)
        
        with tab3:
            st.subheader("Stress Testing")
            
            # Run stress tests based on selected scenarios
            stress_results = {}
            
            if stress_scenarios["Market Crash"]:
                stress_results["Market Crash"] = risk_calculator.stress_test_portfolio(
                    risk_data, scenario_type="market_crash", severity=0.20
                )
            
            if stress_scenarios["Volatility Spike"]:
                stress_results["Volatility Spike"] = risk_calculator.stress_test_portfolio(
                    risk_data, scenario_type="volatility_spike", severity=0.50
                )
            
            if stress_scenarios["Interest Rate Change"]:
                stress_results["Interest Rate Rise"] = risk_calculator.stress_test_portfolio(
                    risk_data, scenario_type="interest_rate", severity=0.02
                )
                stress_results["Interest Rate Fall"] = risk_calculator.stress_test_portfolio(
                    risk_data, scenario_type="interest_rate", severity=-0.02
                )
            
            if stress_scenarios["Commodity Shock"]:
                stress_results["Commodity Rally"] = risk_calculator.stress_test_portfolio(
                    risk_data, scenario_type="commodity_shock", severity=0.30
                )
                stress_results["Commodity Crash"] = risk_calculator.stress_test_portfolio(
                    risk_data, scenario_type="commodity_shock", severity=-0.30
                )
            
            # Display stress test results
            if stress_results:
                stress_df_data = []
                for scenario, result in stress_results.items():
                    stress_df_data.append({
                        'Scenario': scenario,
                        'Portfolio P&L': f"₹{result['total_pnl']:,.2f}",
                        'Portfolio Return': f"{result['portfolio_return']:+.2f}%",
                        'Worst Position': result['worst_position'],
                        'Worst Position Loss': f"₹{result['worst_loss']:,.2f}"
                    })
                
                df_stress = pd.DataFrame(stress_df_data)
                
                # Color code the results
                def color_stress_results(val):
                    if 'P&L' in str(val) or 'Loss' in str(val):
                        if '₹' in str(val):
                            try:
                                num_val = float(str(val).replace('₹', '').replace(',', ''))
                                return 'color: green' if num_val >= 0 else 'color: red'
                            except:
                                return ''
                    elif 'Return' in str(val):
                        if '%' in str(val):
                            try:
                                num_val = float(str(val).replace('%', '').replace('+', ''))
                                return 'color: green' if num_val >= 0 else 'color: red'
                            except:
                                return ''
                    return ''
                
                styled_stress_df = df_stress.style.applymap(color_stress_results)
                st.dataframe(styled_stress_df, use_container_width=True)
                
                # Stress test visualization
                scenarios = list(stress_results.keys())
                pnl_values = [stress_results[s]['total_pnl'] for s in scenarios]
                
                fig_stress = go.Figure()
                fig_stress.add_trace(go.Bar(
                    x=scenarios,
                    y=pnl_values,
                    marker_color=['red' if p < 0 else 'green' for p in pnl_values]
                ))
                
                fig_stress.update_layout(
                    title="Stress Test Results - Portfolio P&L Impact",
                    xaxis_title="Scenario",
                    yaxis_title="P&L Impact (₹)",
                    height=400
                )
                st.plotly_chart(fig_stress, use_container_width=True)
        
        with tab4:
            st.subheader("Scenario Analysis")
            
            # Custom scenario testing
            st.markdown("#### Custom Scenario Parameters")
            
            scenario_col1, scenario_col2 = st.columns(2)
            
            with scenario_col1:
                custom_price_change = st.slider(
                    "Underlying Price Change (%)",
                    min_value=-50.0,
                    max_value=50.0,
                    value=0.0,
                    step=1.0
                )
                
                custom_vol_change = st.slider(
                    "Volatility Change (%)",
                    min_value=-50.0,
                    max_value=100.0,
                    value=0.0,
                    step=5.0
                )
            
            with scenario_col2:
                custom_time_decay = st.slider(
                    "Days Forward",
                    min_value=0,
                    max_value=30,
                    value=0,
                    step=1
                )
                
                custom_rate_change = st.slider(
                    "Interest Rate Change (%)",
                    min_value=-3.0,
                    max_value=3.0,
                    value=0.0,
                    step=0.1
                )
            
            if st.button("Run Custom Scenario"):
                custom_result = risk_calculator.custom_scenario_analysis(
                    risk_data,
                    price_change=custom_price_change/100,
                    vol_change=custom_vol_change/100,
                    time_decay_days=custom_time_decay,
                    rate_change=custom_rate_change/100
                )
                
                st.success(f"**Scenario Result:** Portfolio P&L would be ₹{custom_result['total_pnl']:,.2f} ({custom_result['portfolio_return']:+.2f}%)")
                
                # Show position-by-position impact
                if 'position_impacts' in custom_result:
                    impact_data = []
                    for impact in custom_result['position_impacts']:
                        impact_data.append({
                            'Position': impact['position_name'],
                            'Current Value': f"₹{impact['current_value']:,.2f}",
                            'Scenario Value': f"₹{impact['scenario_value']:,.2f}",
                            'P&L Impact': f"₹{impact['pnl_impact']:,.2f}"
                        })
                    
                    if impact_data:
                        df_impact = pd.DataFrame(impact_data)
                        st.dataframe(df_impact, use_container_width=True)
        
        with tab5:
            st.subheader("Risk Limits Monitoring")
            
            # Check risk limits
            limit_checks = risk_calculator.check_risk_limits(
                risk_data, total_margin, 
                max_portfolio_risk, max_single_position
            )
            
            # Display limit status
            st.markdown("#### Current Limit Status")
            
            for check in limit_checks:
                status_icon = "✅" if check['status'] == 'OK' else "⚠️" if check['status'] == 'WARNING' else "❌"
                status_color = "success" if check['status'] == 'OK' else "warning" if check['status'] == 'WARNING' else "error"
                
                with st.container():
                    if check['status'] == 'OK':
                        st.success(f"{status_icon} **{check['limit_name']}**: {check['message']}")
                    elif check['status'] == 'WARNING':
                        st.warning(f"{status_icon} **{check['limit_name']}**: {check['message']}")
                    else:
                        st.error(f"{status_icon} **{check['limit_name']}**: {check['message']}")
            
            # Risk monitoring settings
            st.markdown("#### Risk Monitoring Alerts")
            
            alert_col1, alert_col2 = st.columns(2)
            
            with alert_col1:
                enable_pnl_alerts = st.checkbox("P&L Alert Threshold", value=True)
                if enable_pnl_alerts:
                    pnl_threshold = st.number_input(
                        "Alert when daily loss exceeds (₹)",
                        min_value=1000,
                        max_value=100000,
                        value=10000,
                        step=1000
                    )
            
            with alert_col2:
                enable_delta_alerts = st.checkbox("Delta Exposure Alert", value=True)
                if enable_delta_alerts:
                    delta_threshold = st.number_input(
                        "Alert when portfolio delta exceeds",
                        min_value=10,
                        max_value=500,
                        value=100,
                        step=10
                    )
            
            # Risk summary
            st.markdown("#### Risk Summary")
            
            risk_summary = risk_calculator.generate_risk_summary(risk_data, total_margin, portfolio_pnl)
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.metric("Overall Risk Score", risk_summary['risk_score'])
                st.caption(risk_summary['risk_description'])
            
            with summary_col2:
                st.metric("Diversification Score", risk_summary['diversification_score'])
                st.caption("Based on commodity and strategy mix")
            
            with summary_col3:
                st.metric("Liquidity Score", risk_summary['liquidity_score'])
                st.caption("Based on position sizes and market depth")

# Risk alerts and notifications
st.subheader("🚨 Risk Alerts")

if active_positions:
    # Check for critical risk alerts
    alerts = []
    
    # Check portfolio P&L
    if portfolio_pnl < -total_margin * 0.1:  # 10% portfolio loss
        alerts.append("🔴 **Critical**: Portfolio loss exceeds 10% of capital")
    
    # Check expiry alerts
    for position in active_positions:
        expiry_date = datetime.strptime(position['expiry_date'], '%Y-%m-%d')
        days_to_expiry = (expiry_date - datetime.now()).days
        
        if days_to_expiry <= 3:
            alerts.append(f"⚠️ **Expiry Alert**: {position['commodity']} {position['strike_price']} {position['option_type']} expires in {days_to_expiry} days")
    
    # Display alerts
    if alerts:
        for alert in alerts:
            if "Critical" in alert:
                st.error(alert)
            else:
                st.warning(alert)
    else:
        st.success("✅ No critical risk alerts at this time")

else:
    st.info("No active positions to monitor for risk alerts")

# Market status and auto-refresh
market_open = is_market_open()
indian_time = get_indian_time()

col1, col2 = st.columns(2)

with col1:
    if market_open:
        st.success(f"🟢 Markets are OPEN | {indian_time.strftime('%H:%M:%S IST')}")
    else:
        st.info(f"🔴 Markets are CLOSED | {indian_time.strftime('%H:%M:%S IST')}")

with col2:
    if st.button("🔄 Refresh Risk Data"):
        st.rerun()

# Auto-refresh during market hours
if st.sidebar.checkbox("Auto Refresh (60s)", value=False) and market_open:
    import time
    time.sleep(60)
    st.rerun()
