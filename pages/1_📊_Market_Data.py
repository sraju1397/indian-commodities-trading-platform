"""
Market Data page for real-time commodity prices and analysis
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from utils.data_fetcher import get_commodity_data, get_indian_commodities
from utils.indian_market_utils import get_indian_time, is_market_open, get_market_status
from utils.logger import logger
from config import Config

# Page configuration
st.set_page_config(page_title="Market Data", page_icon="📊", layout="wide")

# Load custom CSS
with open('static/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_commodity_data(symbol, period, interval):
    """Load commodity data with caching"""
    try:
        return get_commodity_data(symbol, period=period, interval=interval)
    except Exception as e:
        logger.error(f"Error loading commodity data: {str(e)}")
        return None

def main():
    try:
        # Header
        st.markdown("""
            <div class="header-banner" style="background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                url('https://images.pexels.com/photos/159888/pexels-photo-159888.jpeg');">
                <h1>📊 Real-time Market Data</h1>
                <p>Live MCX commodity prices and technical analysis</p>
            </div>
        """, unsafe_allow_html=True)

        # MCX Authentication Status
        api_key = Config.KITE_API_KEY
        access_token = st.session_state.get('kite_access_token') or Config.KITE_ACCESS_TOKEN

        if access_token and access_token != "your_access_token_here":
            st.markdown("""
                <div class="alert alert-success">
                    ✅ Connected to live MCX data via Kite Connect
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="alert alert-warning">
                    ⚠️ Kite Connect authentication required for live MCX data. 
                    Please authenticate in the Kite Auth page.
                </div>
            """, unsafe_allow_html=True)

        # Sidebar controls
        st.sidebar.markdown("""
            <div class="metric-card">
                <h3>Data Controls</h3>
            </div>
        """, unsafe_allow_html=True)

        # Commodity selection
        indian_commodities = get_indian_commodities()
        selected_commodity = st.sidebar.selectbox(
            "Select Commodity",
            options=list(indian_commodities.keys()),
            index=0
        )

        # Time period selection
        time_periods = {
            "1 Day": "1d",
            "5 Days": "5d", 
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y"
        }

        selected_period = st.sidebar.selectbox(
            "Select Time Period",
            options=list(time_periods.keys()),
            index=0
        )

        # Interval selection
        intervals = {
            "1 Minute": "1m",
            "5 Minutes": "5m",
            "15 Minutes": "15m",
            "1 Hour": "1h",
            "1 Day": "1d"
        }

        selected_interval = st.sidebar.selectbox(
            "Select Interval",
            options=list(intervals.keys()),
            index=1 if selected_period in ["1 Day", "5 Days"] else 4
        )

        # Auto-refresh toggle with modern styling
        st.sidebar.markdown("""
            <div class="metric-card">
                <h4>Auto Refresh</h4>
            </div>
        """, unsafe_allow_html=True)
        
        auto_refresh = st.sidebar.checkbox("Enable Auto Refresh (30s)", value=False)

        if auto_refresh:
            st.sidebar.markdown("""
                <div class="alert alert-success">
                    ✅ Auto refresh enabled
                </div>
            """, unsafe_allow_html=True)
            st.rerun()

        # Main content
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>{selected_commodity} Price Chart</h3>
                </div>
            """, unsafe_allow_html=True)

            # Fetch and display data
            with st.spinner("Loading market data..."):
                try:
                    symbol = indian_commodities[selected_commodity]
                    data = load_commodity_data(
                        symbol,
                        period=time_periods[selected_period],
                        interval=intervals[selected_interval]
                    )

                    if data is not None and not data.empty:
                        # Create candlestick chart with modern theme
                        fig = go.Figure(data=go.Candlestick(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name=selected_commodity
                        ))

                        # Add volume as subplot
                        fig.add_trace(
                            go.Bar(
                                x=data.index,
                                y=data['Volume'],
                                name='Volume',
                                yaxis='y2',
                                marker=dict(color='rgba(0,100,80,0.3)')
                            )
                        )

                        # Update layout with modern theme
                        fig.update_layout(
                            template='plotly_dark',
                            title=f"{selected_commodity} - {selected_period} Chart",
                            xaxis_title="Time",
                            yaxis_title="Price (₹)",
                            yaxis2=dict(
                                title="Volume",
                                overlaying='y',
                                side='right'
                            ),
                            height=600,
                            showlegend=True,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=50, r=50, t=50, b=50)
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Technical indicators
                        st.markdown("""
                            <div class="metric-card">
                                <h3>Technical Indicators</h3>
                            </div>
                        """, unsafe_allow_html=True)

                        # Calculate indicators
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

                        # Display indicators in modern cards
                        tech_cols = st.columns(4)

                        with tech_cols[0]:
                            current_price = data['Close'].iloc[-1]
                            sma_20 = data['SMA_20'].iloc[-1]
                            signal = "🟢 Bullish" if current_price > sma_20 else "🔴 Bearish"
                            signal_class = "indicator-bullish" if current_price > sma_20 else "indicator-bearish"
                            
                            st.markdown(f"""
                                <div class="metric-card">
                                    <h4>SMA(20) Signal</h4>
                                    <div class="indicator {signal_class}">
                                        {signal}
                                    </div>
                                    <p>Value: ₹{sma_20:.2f}</p>
                                </div>
                            """, unsafe_allow_html=True)

                        with tech_cols[1]:
                            rsi = data['RSI'].iloc[-1]
                            if rsi > 70:
                                rsi_signal = "🔴 Overbought"
                                rsi_class = "indicator-bearish"
                            elif rsi < 30:
                                rsi_signal = "🟢 Oversold"
                                rsi_class = "indicator-bullish"
                            else:
                                rsi_signal = "🟡 Neutral"
                                rsi_class = "indicator-neutral"

                            st.markdown(f"""
                                <div class="metric-card">
                                    <h4>RSI Signal</h4>
                                    <div class="indicator {rsi_class}">
                                        {rsi_signal}
                                    </div>
                                    <p>Value: {rsi:.2f}</p>
                                </div>
                            """, unsafe_allow_html=True)

                        with tech_cols[2]:
                            macd = data['MACD'].iloc[-1]
                            signal_line = data['Signal'].iloc[-1]
                            macd_signal = "🟢 Bullish" if macd > signal_line else "🔴 Bearish"
                            macd_class = "indicator-bullish" if macd > signal_line else "indicator-bearish"

                            st.markdown(f"""
                                <div class="metric-card">
                                    <h4>MACD Signal</h4>
                                    <div class="indicator {macd_class}">
                                        {macd_signal}
                                    </div>
                                    <p>Value: {macd:.4f}</p>
                                </div>
                            """, unsafe_allow_html=True)

                        with tech_cols[3]:
                            volume = data['Volume'].iloc[-1]
                            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                            volume_signal = "🟢 High" if volume > avg_volume else "🟡 Normal"
                            volume_class = "indicator-bullish" if volume > avg_volume else "indicator-neutral"

                            st.markdown(f"""
                                <div class="metric-card">
                                    <h4>Volume Signal</h4>
                                    <div class="indicator {volume_class}">
                                        {volume_signal}
                                    </div>
                                    <p>Value: {volume:,.0f}</p>
                                </div>
                            """, unsafe_allow_html=True)

                    else:
                        st.error("No data available for the selected commodity and time period.")

                except Exception as e:
                    logger.error(f"Error processing market data: {str(e)}")
                    st.error("Error processing market data. Please try again later.")

        with col2:
            # Market Information
            st.markdown("""
                <div class="metric-card">
                    <h3>Market Information</h3>
                </div>
            """, unsafe_allow_html=True)

            market_open = is_market_open()
            indian_time = get_indian_time()

            status_class = "market-open" if market_open else "market-closed"
            st.markdown(f"""
                <div class="metric-card">
                    <p><strong>Current Time (IST):</strong><br/>{indian_time.strftime('%H:%M:%S')}</p>
                    <div class="market-status {status_class}">
                        {get_market_status()}
                    </div>
                    <p><strong>Next Session:</strong><br/>
                    {'Evening (17:00)' if indian_time.hour < 17 else 'Tomorrow (09:00)'}</p>
                </div>
            """, unsafe_allow_html=True)

            # Quick stats
            if 'data' in locals() and data is not None and not data.empty:
                st.markdown("""
                    <div class="metric-card">
                        <h3>Quick Statistics</h3>
                    </div>
                """, unsafe_allow_html=True)

                current_price = data['Close'].iloc[-1]
                prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100

                day_high = data['High'].iloc[-1] if selected_period == "1 Day" else data['High'].max()
                day_low = data['Low'].iloc[-1] if selected_period == "1 Day" else data['Low'].min()

                # Display metrics in modern cards
                metrics = [
                    ("Current Price", f"₹{current_price:.2f}", f"{change_pct:+.2f}%"),
                    ("Day High", f"₹{day_high:.2f}", None),
                    ("Day Low", f"₹{day_low:.2f}", None),
                    ("Volume", f"{data['Volume'].iloc[-1]:,.0f}", None)
                ]

                for label, value, delta in metrics:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h4>{label}</h4>
                            <p style="font-size: 1.2em;">{value}</p>
                            {f'<p style="color: {"#28a745" if float(delta.strip("% +")) > 0 else "#dc3545"}">{delta}</p>' if delta else ''}
                        </div>
                    """, unsafe_allow_html=True)

                # Price range indicator
                price_range = day_high - day_low
                current_position = (current_price - day_low) / price_range if price_range > 0 else 0.5

                st.markdown("""
                    <div class="metric-card">
                        <h4>Price Position</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                st.progress(current_position)
                st.caption(f"Current price is {current_position*100:.1f}% within today's range")

        # Market Overview Table
        st.markdown("""
            <div class="metric-card">
                <h3>📋 Market Overview - All Commodities</h3>
            </div>
        """, unsafe_allow_html=True)

        try:
            overview_data = []
            for name, symbol in indian_commodities.items():
                try:
                    commodity_data = get_commodity_data(symbol, period='1d')
                    if commodity_data is not None and not commodity_data.empty:
                        current = commodity_data['Close'].iloc[-1]
                        prev_close = commodity_data['Close'].iloc[-2] if len(commodity_data) > 1 else current
                        change = current - prev_close
                        change_pct = (change / prev_close) * 100
                        volume = commodity_data['Volume'].iloc[-1]

                        overview_data.append({
                            'Commodity': name,
                            'Price (₹)': f"{current:.2f}",
                            'Change': f"{change:+.2f}",
                            'Change %': f"{change_pct:+.2f}%",
                            'Volume': f"{volume:,.0f}"
                        })
                except Exception as e:
                    logger.error(f"Error processing overview data for {name}: {str(e)}")
                    continue

            if overview_data:
                df_overview = pd.DataFrame(overview_data)
                st.markdown("""
                    <div class="table-container">
                """, unsafe_allow_html=True)
                st.dataframe(df_overview, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("Unable to fetch overview data at this time.")

        except Exception as e:
            logger.error(f"Error creating market overview: {str(e)}")
            st.error("Error creating market overview. Please try again later.")

        # Export data option
        if st.button("📥 Export Current Data"):
            if 'data' in locals() and data is not None:
                csv = data.to_csv()
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{selected_commodity}_{selected_period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    except Exception as e:
        logger.error(f"Error in Market Data page: {str(e)}")
        st.error("An unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    main()
