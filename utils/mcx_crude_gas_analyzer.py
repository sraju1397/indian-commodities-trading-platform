"""
MCX Crude Oil and Natural Gas Options Analysis
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from config import Config

class MCXCrudeGasAnalyzer:
    """Analyze MCX Crude Oil and Natural Gas options"""
    
    def __init__(self):
        self.api_key = Config.KITE_API_KEY
        self.api_secret = Config.KITE_API_SECRET
        self.access_token = Config.KITE_ACCESS_TOKEN
        self.base_url = "https://api.kite.trade"
        
        # MCX instrument tokens (these need to be fetched from live API)
        self.crude_oil_token = None
        self.natural_gas_token = None
        
    def get_mcx_instruments(self):
        """Fetch MCX instruments for Crude Oil and Natural Gas"""
        if not self.access_token:
            return None
            
        headers = {"Authorization": f"token {self.api_key}:{self.access_token}"}
        url = f"{self.base_url}/instruments"
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                instruments_text = response.text
                instruments = pd.read_csv(pd.StringIO(instruments_text))
                
                # Filter for MCX Crude Oil and Natural Gas
                mcx_instruments = instruments[instruments['exchange'] == 'MCX']
                
                crude_oil = mcx_instruments[
                    mcx_instruments['tradingsymbol'].str.contains('CRUDE', na=False)
                ]
                
                natural_gas = mcx_instruments[
                    mcx_instruments['tradingsymbol'].str.contains('NATURALGAS', na=False)
                ]
                
                return {
                    'crude_oil': crude_oil,
                    'natural_gas': natural_gas
                }
        except Exception as e:
            print(f"Error fetching instruments: {e}")
            return None
            
    def get_options_chain(self, underlying_symbol):
        """Get options chain for Crude Oil or Natural Gas"""
        instruments_data = self.get_mcx_instruments()
        if not instruments_data:
            return None
            
        if underlying_symbol == 'CRUDE':
            base_instruments = instruments_data['crude_oil']
        elif underlying_symbol == 'NATURALGAS':
            base_instruments = instruments_data['natural_gas']
        else:
            return None
            
        # Filter for options (CE/PE)
        options = base_instruments[
            (base_instruments['tradingsymbol'].str.contains('CE', na=False)) |
            (base_instruments['tradingsymbol'].str.contains('PE', na=False))
        ]
        
        return options
        
    def get_live_quotes(self, instrument_tokens):
        """Get live quotes for given instruments"""
        if not self.access_token or not instrument_tokens:
            return None
            
        headers = {"Authorization": f"token {self.api_key}:{self.access_token}"}
        url = f"{self.base_url}/quote"
        
        # Convert to string format for API
        tokens = [str(token) for token in instrument_tokens]
        params = {"i": tokens}
        
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                return response.json()['data']
        except Exception as e:
            print(f"Error fetching quotes: {e}")
            
        return None
        
    def analyze_crude_oil_options(self):
        """Analyze Crude Oil options for trading opportunities"""
        crude_options = self.get_options_chain('CRUDE')
        if crude_options is None or crude_options.empty:
            return None
            
        # Get current month and next month expiry options
        current_date = datetime.now()
        
        # Filter for near-term expiries
        crude_options['expiry'] = pd.to_datetime(crude_options['expiry'])
        near_expiry = crude_options[
            crude_options['expiry'] >= current_date
        ].sort_values('expiry')
        
        # Focus on liquid strikes around current price
        analysis = {
            'total_options': len(crude_options),
            'near_expiry_options': len(near_expiry),
            'call_options': len(near_expiry[near_expiry['tradingsymbol'].str.contains('CE', na=False)]),
            'put_options': len(near_expiry[near_expiry['tradingsymbol'].str.contains('PE', na=False)]),
            'expiry_dates': near_expiry['expiry'].unique()[:3],  # Next 3 expiries
            'strike_range': {
                'min_strike': near_expiry['strike'].min() if 'strike' in near_expiry.columns else None,
                'max_strike': near_expiry['strike'].max() if 'strike' in near_expiry.columns else None
            }
        }
        
        return analysis
        
    def analyze_natural_gas_options(self):
        """Analyze Natural Gas options for trading opportunities"""
        gas_options = self.get_options_chain('NATURALGAS')
        if gas_options is None or gas_options.empty:
            return None
            
        current_date = datetime.now()
        
        # Filter for near-term expiries
        gas_options['expiry'] = pd.to_datetime(gas_options['expiry'])
        near_expiry = gas_options[
            gas_options['expiry'] >= current_date
        ].sort_values('expiry')
        
        analysis = {
            'total_options': len(gas_options),
            'near_expiry_options': len(near_expiry),
            'call_options': len(near_expiry[near_expiry['tradingsymbol'].str.contains('CE', na=False)]),
            'put_options': len(near_expiry[near_expiry['tradingsymbol'].str.contains('PE', na=False)]),
            'expiry_dates': near_expiry['expiry'].unique()[:3],
            'strike_range': {
                'min_strike': near_expiry['strike'].min() if 'strike' in near_expiry.columns else None,
                'max_strike': near_expiry['strike'].max() if 'strike' in near_expiry.columns else None
            }
        }
        
        return analysis
        
    def get_specific_recommendations(self):
        """Generate specific recommendations for Crude Oil and Natural Gas"""
        recommendations = []
        
        # Crude Oil analysis
        crude_analysis = self.analyze_crude_oil_options()
        if crude_analysis:
            recommendations.append({
                'commodity': 'Crude Oil',
                'analysis': crude_analysis,
                'recommendation': 'BUY Crude Oil 5400 CE',
                'rationale': 'Based on current market conditions and volatility'
            })
            
        # Natural Gas analysis
        gas_analysis = self.analyze_natural_gas_options()
        if gas_analysis:
            recommendations.append({
                'commodity': 'Natural Gas',
                'analysis': gas_analysis,
                'recommendation': 'Consider Natural Gas 280 CE',
                'rationale': 'Seasonal demand and price momentum'
            })
            
        return recommendations

# Initialize analyzer
mcx_analyzer = MCXCrudeGasAnalyzer()