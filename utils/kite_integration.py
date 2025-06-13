"""
Kite Connect API integration for live MCX data
"""
import os
import requests
import pandas as pd
from datetime import datetime
import streamlit as st
from config import Config

class KiteConnectAPI:
    """Handle Kite Connect API integration for live MCX data"""
    
    def __init__(self):
        self.api_key = Config.KITE_API_KEY
        self.api_secret = Config.KITE_API_SECRET
        self.access_token = Config.KITE_ACCESS_TOKEN
        self.base_url = "https://api.kite.trade"
        
    def get_login_url(self):
        """Generate Kite Connect login URL"""
        return f"https://kite.trade/connect/login?api_key={self.api_key}"
    
    def generate_session(self, request_token):
        """Generate access token from request token"""
        import hashlib
        
        checksum = hashlib.sha256(f"{self.api_key}{request_token}{self.api_secret}".encode()).hexdigest()
        
        url = f"{self.base_url}/session/token"
        data = {
            "api_key": self.api_key,
            "request_token": request_token,
            "checksum": checksum
        }
        
        response = requests.post(url, data=data)
        if response.status_code == 200:
            session_data = response.json()
            return session_data['data']['access_token']
        return None
    
    def get_instruments(self):
        """Fetch all available instruments"""
        if not self.access_token:
            return None
            
        headers = {
            "Authorization": f"token {self.api_key}:{self.access_token}"
        }
        
        url = f"{self.base_url}/instruments"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return pd.read_csv(response.text)
        return None
    
    def get_mcx_commodities(self):
        """Get MCX commodity instrument tokens"""
        instruments = self.get_instruments()
        if instruments is None:
            return None
            
        # Filter MCX commodities
        mcx_commodities = instruments[
            (instruments['exchange'] == 'MCX') & 
            (instruments['segment'] == 'MCX')
        ]
        
        # Focus on major commodities
        commodity_keywords = ['GOLD', 'SILVER', 'CRUDE', 'NATURALGAS', 'COPPER']
        filtered = mcx_commodities[
            mcx_commodities['tradingsymbol'].str.contains('|'.join(commodity_keywords), na=False)
        ]
        
        return filtered
    
    def get_live_quotes(self, instrument_tokens):
        """Fetch live quotes for given instrument tokens"""
        if not self.access_token:
            return None
            
        headers = {
            "Authorization": f"token {self.api_key}:{self.access_token}"
        }
        
        # Convert tokens to string format
        tokens = [str(token) for token in instrument_tokens]
        url = f"{self.base_url}/quote"
        
        params = {"i": tokens}
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            return response.json()['data']
        return None
    
    def get_historical_data(self, instrument_token, from_date, to_date, interval="day"):
        """Fetch historical data for technical analysis"""
        if not self.access_token:
            return None
            
        headers = {
            "Authorization": f"token {self.api_key}:{self.access_token}"
        }
        
        url = f"{self.base_url}/instruments/historical/{instrument_token}/{interval}"
        params = {
            "from": from_date,
            "to": to_date
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()['data']['candles']
            df = pd.DataFrame(data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            return df
        return None
    
    def get_options_chain(self, instrument_token):
        """Fetch options chain for a commodity"""
        if not self.access_token:
            return None
            
        headers = {
            "Authorization": f"token {self.api_key}:{self.access_token}"
        }
        
        url = f"{self.base_url}/instruments"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            instruments = pd.read_csv(response.text)
            
            # Filter options for the specific commodity
            options = instruments[
                (instruments['instrument_token'] == instrument_token) |
                (instruments['name'].str.contains(str(instrument_token), na=False))
            ]
            
            return options
        return None
    
    def is_authenticated(self):
        """Check if API credentials are valid"""
        return all([self.api_key, self.api_secret, self.access_token])
    
    def test_connection(self):
        """Test the API connection"""
        if not self.access_token:
            return False, "Access token not available"
            
        headers = {
            "Authorization": f"token {self.api_key}:{self.access_token}"
        }
        
        url = f"{self.base_url}/user/profile"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return True, "Connection successful"
        else:
            return False, f"Connection failed: {response.status_code}"

# Global instance
kite_api = KiteConnectAPI()

def get_live_mcx_data():
    """Fetch live MCX commodity data"""
    try:
        # Get MCX instruments
        mcx_instruments = kite_api.get_mcx_commodities()
        if mcx_instruments is None or mcx_instruments.empty:
            return None
            
        # Get live quotes for major commodities
        instrument_tokens = mcx_instruments['instrument_token'].head(10).tolist()
        live_quotes = kite_api.get_live_quotes(instrument_tokens)
        
        if live_quotes:
            # Process and return structured data
            processed_data = []
            for token, quote in live_quotes.items():
                processed_data.append({
                    'instrument_token': token,
                    'last_price': quote.get('last_price', 0),
                    'change': quote.get('net_change', 0),
                    'change_percent': quote.get('percentage_change', 0),
                    'volume': quote.get('volume', 0),
                    'timestamp': datetime.now()
                })
            
            return pd.DataFrame(processed_data)
        
    except Exception as e:
        st.error(f"Error fetching live MCX data: {str(e)}")
        return None

def authenticate_kite_user():
    """Handle Kite Connect authentication flow"""
    st.subheader("Kite Connect Authentication")
    
    if not kite_api.api_key or not kite_api.api_secret:
        st.error("Kite API credentials not configured. Please check your environment variables.")
        return False
    
    if kite_api.access_token:
        # Test existing connection
        success, message = kite_api.test_connection()
        if success:
            st.success("✅ Connected to Kite Connect API")
            return True
        else:
            st.warning(f"Connection test failed: {message}")
    
    # Generate login URL
    login_url = kite_api.get_login_url()
    st.info("Please complete the authentication process:")
    st.markdown(f"[Click here to login to Kite Connect]({login_url})")
    
    # Input for request token
    request_token = st.text_input("Enter the request token from the callback URL:")
    
    if st.button("Generate Access Token") and request_token:
        access_token = kite_api.generate_session(request_token)
        if access_token:
            st.success("✅ Authentication successful!")
            st.info(f"Access Token: {access_token}")
            st.info("Please add this access token to your environment variables as KITE_ACCESS_TOKEN")
            return True
        else:
            st.error("Authentication failed. Please try again.")
    
    return False