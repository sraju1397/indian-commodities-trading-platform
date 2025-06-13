"""
Risk calculation and analysis utilities for options trading
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from utils.options_calculator import black_scholes_price, calculate_greeks

class RiskCalculator:
    """
    Comprehensive risk calculation and analysis for options portfolios
    """
    
    def __init__(self):
        """Initialize risk calculator"""
        pass
    
    def calculate_portfolio_var(self, risk_data, confidence_level=95, time_horizon=1):
        """
        Calculate Value at Risk (VaR) for the portfolio using historical simulation
        
        Args:
            risk_data (list): List of position risk data
            confidence_level (int): Confidence level (90, 95, 99)
            time_horizon (int): Time horizon in days
        
        Returns:
            dict: VaR calculation results
        """
        try:
            if not risk_data:
                return {'Portfolio VaR': 0, 'Expected Shortfall': 0}
            
            # Generate scenarios using historical price movements
            scenarios = self._generate_price_scenarios(risk_data, time_horizon)
            
            portfolio_pnl_scenarios = []
            
            for scenario in scenarios:
                portfolio_pnl = 0
                
                for rd in risk_data:
                    position = rd['position']
                    current_price = rd['current_price']
                    
                    # Apply scenario price change
                    new_price = current_price * scenario['price_multiplier']
                    
                    # Calculate new option price
                    expiry_date = datetime.strptime(position['expiry_date'], '%Y-%m-%d')
                    time_to_expiry = max((expiry_date - datetime.now()).days / 365.0 - time_horizon/365.0, 0.001)
                    
                    new_vol = scenario.get('volatility_multiplier', 1.0) * 0.25  # Assumed base vol
                    
                    new_option_price = black_scholes_price(
                        new_price, 
                        position['strike_price'], 
                        time_to_expiry, 
                        0.065, 
                        new_vol, 
                        position['option_type'].lower()
                    )
                    
                    # Calculate position P&L
                    if position['action'] == 'Buy':
                        position_pnl = (new_option_price - rd['option_price']) * position['quantity']
                    else:
                        position_pnl = (rd['option_price'] - new_option_price) * position['quantity']
                    
                    portfolio_pnl += position_pnl
                
                portfolio_pnl_scenarios.append(portfolio_pnl)
            
            # Calculate VaR and Expected Shortfall
            portfolio_pnl_scenarios = np.array(portfolio_pnl_scenarios)
            var_percentile = 100 - confidence_level
            
            var_value = np.percentile(portfolio_pnl_scenarios, var_percentile)
            
            # Expected Shortfall (Conditional VaR)
            tail_losses = portfolio_pnl_scenarios[portfolio_pnl_scenarios <= var_value]
            expected_shortfall = np.mean(tail_losses) if len(tail_losses) > 0 else var_value
            
            return {
                f'{confidence_level}% Portfolio VaR': var_value,
                'Expected Shortfall': expected_shortfall,
                'Best Case': np.max(portfolio_pnl_scenarios),
                'Worst Case': np.min(portfolio_pnl_scenarios),
                'Mean P&L': np.mean(portfolio_pnl_scenarios),
                'Volatility': np.std(portfolio_pnl_scenarios),
                'var_distribution': portfolio_pnl_scenarios
            }
            
        except Exception as e:
            print(f"Error calculating portfolio VaR: {str(e)}")
            return {'Portfolio VaR': 0, 'Expected Shortfall': 0}
    
    def calculate_position_var(self, risk_data, confidence_level=95, time_horizon=1):
        """
        Calculate VaR for a single position
        
        Args:
            risk_data (dict): Single position risk data
            confidence_level (int): Confidence level
            time_horizon (int): Time horizon in days
        
        Returns:
            float: Position VaR
        """
        try:
            position = risk_data['position']
            current_price = risk_data['current_price']
            option_price = risk_data['option_price']
            
            # Generate price scenarios
            price_scenarios = self._generate_single_asset_scenarios(current_price, time_horizon)
            position_pnl_scenarios = []
            
            for new_price in price_scenarios:
                # Calculate new option price
                expiry_date = datetime.strptime(position['expiry_date'], '%Y-%m-%d')
                time_to_expiry = max((expiry_date - datetime.now()).days / 365.0 - time_horizon/365.0, 0.001)
                
                new_option_price = black_scholes_price(
                    new_price, 
                    position['strike_price'], 
                    time_to_expiry, 
                    0.065, 
                    0.25, 
                    position['option_type'].lower()
                )
                
                # Calculate position P&L
                if position['action'] == 'Buy':
                    position_pnl = (new_option_price - option_price) * position['quantity']
                else:
                    position_pnl = (option_price - new_option_price) * position['quantity']
                
                position_pnl_scenarios.append(position_pnl)
            
            var_percentile = 100 - confidence_level
            var_value = np.percentile(position_pnl_scenarios, var_percentile)
            
            return var_value
            
        except Exception as e:
            print(f"Error calculating position VaR: {str(e)}")
            return 0
    
    def calculate_portfolio_greeks(self, risk_data):
        """
        Calculate portfolio-level Greeks
        
        Args:
            risk_data (list): List of position risk data
        
        Returns:
            dict: Portfolio Greeks
        """
        try:
            portfolio_greeks = {
                'total_delta': 0,
                'total_gamma': 0,
                'total_theta': 0,
                'total_vega': 0,
                'total_rho': 0
            }
            
            for rd in risk_data:
                position = rd['position']
                greeks = rd['greeks']
                quantity = position['quantity']
                
                # Adjust for position direction
                multiplier = quantity if position['action'] == 'Buy' else -quantity
                
                portfolio_greeks['total_delta'] += greeks['delta'] * multiplier
                portfolio_greeks['total_gamma'] += greeks['gamma'] * multiplier
                portfolio_greeks['total_theta'] += greeks['theta'] * multiplier
                portfolio_greeks['total_vega'] += greeks['vega'] * multiplier
                portfolio_greeks['total_rho'] += greeks['rho'] * multiplier
            
            return portfolio_greeks
            
        except Exception as e:
            print(f"Error calculating portfolio Greeks: {str(e)}")
            return {'total_delta': 0, 'total_gamma': 0, 'total_theta': 0, 'total_vega': 0, 'total_rho': 0}
    
    def stress_test_portfolio(self, risk_data, scenario_type='market_crash', severity=0.20):
        """
        Perform stress tests on the portfolio
        
        Args:
            risk_data (list): List of position risk data
            scenario_type (str): Type of stress test
            severity (float): Severity of the stress scenario
        
        Returns:
            dict: Stress test results
        """
        try:
            total_pnl = 0
            position_impacts = []
            
            for rd in risk_data:
                position = rd['position']
                current_price = rd['current_price']
                option_price = rd['option_price']
                
                # Apply stress scenario
                if scenario_type == 'market_crash':
                    new_price = current_price * (1 - severity)
                    new_vol = 0.25 * 1.5  # Volatility spike during crash
                elif scenario_type == 'market_rally':
                    new_price = current_price * (1 + severity)
                    new_vol = 0.25 * 0.8  # Volatility crush during rally
                elif scenario_type == 'volatility_spike':
                    new_price = current_price  # Price unchanged
                    new_vol = 0.25 * (1 + severity)
                elif scenario_type == 'volatility_crush':
                    new_price = current_price
                    new_vol = 0.25 * (1 - severity)
                elif scenario_type == 'commodity_shock':
                    new_price = current_price * (1 + severity)
                    new_vol = 0.25 * 1.3
                elif scenario_type == 'interest_rate':
                    new_price = current_price
                    new_vol = 0.25
                    # Interest rate changes affect option prices through rho
                else:
                    new_price = current_price
                    new_vol = 0.25
                
                # Calculate new option price
                expiry_date = datetime.strptime(position['expiry_date'], '%Y-%m-%d')
                time_to_expiry = max((expiry_date - datetime.now()).days / 365.0, 0.001)
                
                new_rate = 0.065 + severity if scenario_type == 'interest_rate' else 0.065
                
                new_option_price = black_scholes_price(
                    new_price, 
                    position['strike_price'], 
                    time_to_expiry, 
                    new_rate, 
                    new_vol, 
                    position['option_type'].lower()
                )
                
                # Calculate position P&L
                if position['action'] == 'Buy':
                    position_pnl = (new_option_price - option_price) * position['quantity']
                else:
                    position_pnl = (option_price - new_option_price) * position['quantity']
                
                total_pnl += position_pnl
                
                position_impacts.append({
                    'position': f"{position['commodity']} {position['strike_price']} {position['option_type']}",
                    'pnl': position_pnl,
                    'price_change': (new_price - current_price) / current_price * 100,
                    'option_price_change': (new_option_price - option_price) / option_price * 100
                })
            
            # Calculate portfolio metrics
            total_capital = sum([abs(rd['margin']) for rd in risk_data])
            portfolio_return = (total_pnl / total_capital * 100) if total_capital > 0 else 0
            
            # Find worst performing position
            worst_position = min(position_impacts, key=lambda x: x['pnl'])
            
            return {
                'total_pnl': total_pnl,
                'portfolio_return': portfolio_return,
                'worst_position': worst_position['position'],
                'worst_loss': worst_position['pnl'],
                'position_impacts': position_impacts
            }
            
        except Exception as e:
            print(f"Error in stress testing: {str(e)}")
            return {'total_pnl': 0, 'portfolio_return': 0, 'worst_position': 'Unknown', 'worst_loss': 0}
    
    def custom_scenario_analysis(self, risk_data, price_change=0, vol_change=0, 
                                time_decay_days=0, rate_change=0):
        """
        Perform custom scenario analysis
        
        Args:
            risk_data (list): Position risk data
            price_change (float): Price change as decimal (0.1 = 10%)
            vol_change (float): Volatility change as decimal
            time_decay_days (int): Days of time decay
            rate_change (float): Interest rate change as decimal
        
        Returns:
            dict: Scenario analysis results
        """
        try:
            total_pnl = 0
            position_impacts = []
            
            for rd in risk_data:
                position = rd['position']
                current_price = rd['current_price']
                option_price = rd['option_price']
                
                # Apply scenario changes
                new_price = current_price * (1 + price_change)
                new_vol = 0.25 * (1 + vol_change)
                new_rate = 0.065 + rate_change
                
                # Calculate new time to expiry
                expiry_date = datetime.strptime(position['expiry_date'], '%Y-%m-%d')
                original_tte = (expiry_date - datetime.now()).days / 365.0
                new_tte = max(original_tte - time_decay_days/365.0, 0.001)
                
                # Calculate new option price
                new_option_price = black_scholes_price(
                    new_price, 
                    position['strike_price'], 
                    new_tte, 
                    new_rate, 
                    new_vol, 
                    position['option_type'].lower()
                )
                
                # Calculate position P&L
                if position['action'] == 'Buy':
                    position_pnl = (new_option_price - option_price) * position['quantity']
                else:
                    position_pnl = (option_price - new_option_price) * position['quantity']
                
                total_pnl += position_pnl
                
                position_impacts.append({
                    'position_name': f"{position['commodity']} {position['strike_price']} {position['option_type']}",
                    'current_value': option_price * position['quantity'],
                    'scenario_value': new_option_price * position['quantity'],
                    'pnl_impact': position_pnl
                })
            
            # Calculate portfolio return
            total_capital = sum([abs(rd['margin']) for rd in risk_data])
            portfolio_return = (total_pnl / total_capital * 100) if total_capital > 0 else 0
            
            return {
                'total_pnl': total_pnl,
                'portfolio_return': portfolio_return,
                'position_impacts': position_impacts
            }
            
        except Exception as e:
            print(f"Error in custom scenario analysis: {str(e)}")
            return {'total_pnl': 0, 'portfolio_return': 0, 'position_impacts': []}
    
    def check_risk_limits(self, risk_data, total_margin, max_portfolio_risk_pct, max_single_position_pct):
        """
        Check if portfolio violates risk limits
        
        Args:
            risk_data (list): Position risk data
            total_margin (float): Total portfolio margin
            max_portfolio_risk_pct (float): Maximum portfolio risk percentage
            max_single_position_pct (float): Maximum single position percentage
        
        Returns:
            list: List of limit check results
        """
        try:
            limit_checks = []
            
            # Portfolio-level risk check
            portfolio_var = self.calculate_portfolio_var(risk_data, confidence_level=95, time_horizon=1)
            var_amount = abs(portfolio_var.get('95% Portfolio VaR', 0))
            var_percentage = (var_amount / total_margin * 100) if total_margin > 0 else 0
            
            if var_percentage > max_portfolio_risk_pct:
                limit_checks.append({
                    'limit_name': 'Portfolio VaR Limit',
                    'status': 'VIOLATION',
                    'message': f'Portfolio VaR ({var_percentage:.1f}%) exceeds limit ({max_portfolio_risk_pct:.1f}%)',
                    'current_value': var_percentage,
                    'limit_value': max_portfolio_risk_pct
                })
            else:
                limit_checks.append({
                    'limit_name': 'Portfolio VaR Limit',
                    'status': 'OK',
                    'message': f'Portfolio VaR ({var_percentage:.1f}%) within limit ({max_portfolio_risk_pct:.1f}%)',
                    'current_value': var_percentage,
                    'limit_value': max_portfolio_risk_pct
                })
            
            # Single position risk checks
            for rd in risk_data:
                position = rd['position']
                position_margin = abs(rd['margin'])
                position_percentage = (position_margin / total_margin * 100) if total_margin > 0 else 0
                
                position_name = f"{position['commodity']} {position['strike_price']} {position['option_type']}"
                
                if position_percentage > max_single_position_pct:
                    limit_checks.append({
                        'limit_name': f'Single Position Limit ({position_name})',
                        'status': 'VIOLATION',
                        'message': f'Position size ({position_percentage:.1f}%) exceeds limit ({max_single_position_pct:.1f}%)',
                        'current_value': position_percentage,
                        'limit_value': max_single_position_pct
                    })
            
            # Greeks limits
            portfolio_greeks = self.calculate_portfolio_greeks(risk_data)
            delta_exposure = abs(portfolio_greeks['total_delta'])
            
            # Delta limit (example: no more than 100 delta exposure)
            delta_limit = 100
            if delta_exposure > delta_limit:
                limit_checks.append({
                    'limit_name': 'Delta Exposure Limit',
                    'status': 'WARNING',
                    'message': f'Delta exposure ({delta_exposure:.1f}) exceeds recommended limit ({delta_limit})',
                    'current_value': delta_exposure,
                    'limit_value': delta_limit
                })
            
            # Theta limit (example: no more than 5% of portfolio value per day)
            daily_theta = abs(portfolio_greeks['total_theta'])
            theta_limit_pct = 5.0
            theta_percentage = (daily_theta / total_margin * 100) if total_margin > 0 else 0
            
            if theta_percentage > theta_limit_pct:
                limit_checks.append({
                    'limit_name': 'Daily Theta Limit',
                    'status': 'WARNING',
                    'message': f'Daily theta decay ({theta_percentage:.1f}%) exceeds limit ({theta_limit_pct:.1f}%)',
                    'current_value': theta_percentage,
                    'limit_value': theta_limit_pct
                })
            
            return limit_checks
            
        except Exception as e:
            print(f"Error checking risk limits: {str(e)}")
            return []
    
    def generate_risk_summary(self, risk_data, total_margin, current_pnl):
        """
        Generate overall risk summary and scores
        
        Args:
            risk_data (list): Position risk data
            total_margin (float): Total portfolio margin
            current_pnl (float): Current portfolio P&L
        
        Returns:
            dict: Risk summary with scores
        """
        try:
            # Calculate risk metrics
            portfolio_var = self.calculate_portfolio_var(risk_data)
            var_amount = abs(portfolio_var.get('95% Portfolio VaR', 0))
            var_percentage = (var_amount / total_margin * 100) if total_margin > 0 else 0
            
            portfolio_greeks = self.calculate_portfolio_greeks(risk_data)
            
            # Risk score calculation (0-100, lower is better)
            risk_score = 0
            
            # VaR component (0-40 points)
            if var_percentage > 10:
                risk_score += 40
            elif var_percentage > 5:
                risk_score += 20 + (var_percentage - 5) * 4
            else:
                risk_score += var_percentage * 4
            
            # Greeks component (0-30 points)
            delta_exposure = abs(portfolio_greeks['total_delta'])
            if delta_exposure > 150:
                risk_score += 30
            elif delta_exposure > 100:
                risk_score += 15 + (delta_exposure - 100) * 0.3
            else:
                risk_score += delta_exposure * 0.15
            
            # Concentration component (0-30 points)
            commodity_exposure = {}
            for rd in risk_data:
                commodity = rd['position']['commodity']
                if commodity not in commodity_exposure:
                    commodity_exposure[commodity] = 0
                commodity_exposure[commodity] += abs(rd['margin'])
            
            max_concentration = max(commodity_exposure.values()) / total_margin * 100 if commodity_exposure and total_margin > 0 else 0
            
            if max_concentration > 50:
                risk_score += 30
            elif max_concentration > 30:
                risk_score += 15 + (max_concentration - 30) * 0.75
            else:
                risk_score += max_concentration * 0.5
            
            # Risk level description
            if risk_score < 30:
                risk_level = "🟢 Low Risk"
                risk_description = "Portfolio risk is well-managed"
            elif risk_score < 60:
                risk_level = "🟡 Moderate Risk"
                risk_description = "Portfolio has moderate risk exposure"
            else:
                risk_level = "🔴 High Risk"
                risk_description = "Portfolio has high risk exposure"
            
            # Diversification score
            num_commodities = len(commodity_exposure)
            diversification_score = min(100, num_commodities * 20)  # Max 100 for 5+ commodities
            
            # Liquidity score (simplified based on position sizes)
            avg_position_size = np.mean([abs(rd['margin']) for rd in risk_data]) if risk_data else 0
            if avg_position_size < total_margin * 0.05:  # Small positions
                liquidity_score = 90
            elif avg_position_size < total_margin * 0.1:
                liquidity_score = 70
            else:
                liquidity_score = 50
            
            return {
                'risk_score': f"{risk_score:.0f}/100",
                'risk_level': risk_level,
                'risk_description': risk_description,
                'diversification_score': f"{diversification_score:.0f}/100",
                'liquidity_score': f"{liquidity_score:.0f}/100",
                'var_percentage': var_percentage,
                'max_concentration': max_concentration,
                'delta_exposure': delta_exposure
            }
            
        except Exception as e:
            print(f"Error generating risk summary: {str(e)}")
            return {
                'risk_score': "0/100",
                'risk_level': "Unknown",
                'risk_description': "Unable to calculate risk",
                'diversification_score': "0/100",
                'liquidity_score': "0/100"
            }
    
    def _generate_price_scenarios(self, risk_data, time_horizon, num_scenarios=1000):
        """Generate price scenarios for VaR calculation"""
        try:
            scenarios = []
            
            # Use historical volatility to generate scenarios
            np.random.seed(42)
            
            for _ in range(num_scenarios):
                # Generate correlated price movements
                price_shock = np.random.normal(0, 0.02 * np.sqrt(time_horizon))  # 2% daily vol
                vol_shock = np.random.normal(1, 0.1)  # 10% vol of vol
                
                scenario = {
                    'price_multiplier': 1 + price_shock,
                    'volatility_multiplier': max(0.1, vol_shock)  # Prevent negative vol
                }
                scenarios.append(scenario)
            
            return scenarios
            
        except Exception as e:
            print(f"Error generating price scenarios: {str(e)}")
            return [{'price_multiplier': 1, 'volatility_multiplier': 1}] * 100
    
    def _generate_single_asset_scenarios(self, current_price, time_horizon, num_scenarios=1000):
        """Generate price scenarios for single asset"""
        try:
            np.random.seed(42)
            daily_vol = 0.02  # 2% daily volatility
            
            scenarios = []
            for _ in range(num_scenarios):
                shock = np.random.normal(0, daily_vol * np.sqrt(time_horizon))
                new_price = current_price * (1 + shock)
                scenarios.append(max(new_price, current_price * 0.1))  # Floor at 10% of current price
            
            return scenarios
            
        except Exception as e:
            print(f"Error generating single asset scenarios: {str(e)}")
            return [current_price] * 100
