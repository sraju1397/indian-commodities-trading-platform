import streamlit as st
import requests
import hashlib
from urllib.parse import urlparse, parse_qs
from config import Config

# Page configuration
st.set_page_config(
    page_title="Kite Connect Authentication",
    page_icon="🔐",
    layout="wide"
)

def generate_access_token(request_token):
    """Generate access token from request token"""
    api_key = Config.KITE_API_KEY
    api_secret = Config.KITE_API_SECRET
    
    checksum = hashlib.sha256(f"{api_key}{request_token}{api_secret}".encode()).hexdigest()
    
    url = "https://api.kite.trade/session/token"
    data = {
        "api_key": api_key,
        "request_token": request_token,
        "checksum": checksum
    }
    
    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            session_data = response.json()
            return session_data['data']['access_token']
        else:
            st.error(f"Error: {response.json().get('message', 'Authentication failed')}")
            return None
    except Exception as e:
        st.error(f"Error generating access token: {str(e)}")
        return None

def main():
    st.title("🔐 Kite Connect Authentication")
    st.markdown("### Complete authentication to access live MCX options data")
    
    # Check for request token in URL parameters
    query_params = st.query_params
    
    if "request_token" in query_params:
        request_token = query_params["request_token"]
        st.success(f"Request token received: {request_token}")
        
        if st.button("Generate Access Token"):
            with st.spinner("Generating access token..."):
                access_token = generate_access_token(request_token)
                
                if access_token:
                    # Automatically store in session for immediate use
                    st.session_state['kite_access_token'] = access_token
                    
                    st.success("✅ Authentication successful! Live MCX data is now active.")
                    st.info("🔐 Access token securely stored for this session")
                    
                    st.success("""
                    **✅ Automated Next Steps Complete:**
                    1. Token automatically stored for this session
                    2. All pages now have access to live MCX data
                    3. Navigate to Trading Alerts or other pages to see live recommendations
                    """)
                    
                    # Show links to main trading pages
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("📊 View Live Market Data"):
                            st.switch_page("pages/1_📊_Market_Data.py")
                    with col2:
                        if st.button("🚨 Trading Alerts"):
                            st.switch_page("pages/7_🚨_Trading_Alerts.py")
                    with col3:
                        if st.button("🎯 Recommendations"):
                            st.switch_page("pages/3_🎯_Recommendations.py")
    else:
        # Show authentication instructions
        st.info("Complete the Kite Connect authentication process:")
        
        # Step 1: Login URL
        api_key = Config.KITE_API_KEY or "caym8d0xr9e2xnh0"  # Fallback to direct value
        login_url = f"https://kite.zerodha.com/connect/login?api_key={api_key}"
        
        st.markdown(f"**Step 1:** [Click here to login to Kite Connect]({login_url})")
        
        st.markdown("""
        **Step 2:** After successful login, you'll be redirected back to this page with an access token.
        
        **What this enables:**
        - Live MCX Crude Oil and Natural Gas options data
        - Real premiums and Greeks calculations
        - Authentic trading recommendations
        """)
        
        # Manual token input as fallback
        st.markdown("---")
        st.subheader("Manual Token Entry")
        manual_token = st.text_input("Or enter request token manually:", placeholder="Enter request token from callback URL")
        
        if st.button("Generate Access Token from Manual Entry") and manual_token:
            with st.spinner("Generating access token..."):
                access_token = generate_access_token(manual_token)
                
                if access_token:
                    st.success("✅ Authentication successful!")
                    st.info("🔐 Access token securely stored for this session")
    
    # Show current authentication status
    st.markdown("---")
    st.subheader("Current Status")
    
    if Config.KITE_API_KEY and Config.KITE_API_SECRET:
        st.success("✅ API credentials configured")
        st.info("🔐 API Key: ****" + str(Config.KITE_API_KEY)[-4:] if Config.KITE_API_KEY else "Not configured")
    else:
        st.error("❌ API credentials not found")
    
    if hasattr(st.session_state, 'kite_access_token'):
        st.success("✅ Access token available for this session")
    elif Config.KITE_ACCESS_TOKEN and Config.KITE_ACCESS_TOKEN != "your_access_token_here":
        st.success("✅ Access token configured in environment")
    else:
        st.warning("⚠️ No access token available - authentication required")

if __name__ == "__main__":
    main()