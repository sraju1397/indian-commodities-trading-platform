zxxzz"""
Options pricing and Greeks calculation utilities for Indian commodities

This module provides comprehensive options pricing and analysis tools including:
- Black-Scholes pricing model
- Greeks calculations (Delta, Gamma, Theta, Vega, Rho)
- Implied volatility calculation
- Strategy payoff analysis
- Synthetic options chain generation
- Monte Carlo simulation with batch processing
"""
from typing import Dict, List, Union, Optional, Tuple, Any
from numpy.typing import NDArray
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd
from datetime import datetime, timedelta
from .logger import logger

# Configure logging
logger.info("Initializing options calculator module")
logger.debug("Loading required mathematical and financial functions")

def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """
    Calculate Black-Scholes option price
    
    Args:
        S: Current price of underlying
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility of underlying
        option_type: 'call' or 'put'
    
    Returns:
        float: Option price
    """
    try:
        logger.debug(f"Calculating {option_type} option price: S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
        
        # Input validation to prevent mathematical errors
        if S <= 0 or K <= 0:
            logger.warning(f"Invalid price inputs: S={S}, K={K}")
            return 0
        
        if T <= 0:
            # Option has expired
            logger.debug("Option has expired, calculating intrinsic value")
            if option_type.lower() == 'call':
                value = max(S - K, 0)
            else:
                value = max(K - S, 0)
            logger.info(f"Expired option intrinsic value: {value}")
            return value
        
        if sigma <= 0:
            logger.warning(f"Invalid volatility {sigma}, using minimum value 0.01")
            sigma = 0.01  # Minimum volatility to avoid division by zero
        
        # Calculate d1 and d2 with error handling
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
        except (ValueError, ZeroDivisionError):
            return 0
        
        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type.lower() == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        price = max(price, 0)  # Price cannot be negative
        logger.debug(f"Calculated {option_type} option price: {price}")
        return price
        
    except Exception as e:
        logger.error(f"Error in Black-Scholes calculation: {str(e)}")
        return 0

def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> dict:
    """
    Calculate option Greeks
    
    Args:
        S: Current price of underlying
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility of underlying
        option_type: 'call' or 'put'
    
    Returns:
        dict: Dictionary containing all Greeks (delta, gamma, theta, vega, rho)
    """
    try:
        logger.debug(f"Calculating Greeks for {option_type} option: S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
        
        if T <= 0:
            # Option has expired - Greeks are zero except Delta
            logger.debug("Option has expired, setting appropriate Greeks values")
            if option_type.lower() == 'call':
                delta = 1.0 if S > K else 0.0
            else:
                delta = -1.0 if S < K else 0.0
            
            logger.info(f"Expired option Greeks - Delta: {delta}")
            
            return {
                'delta': delta,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        if sigma <= 0:
            sigma = 0.01
        if T <= 0:
            T = 0.001  # Minimum time to prevent division by zero
        if S <= 0 or K <= 0:
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Common terms
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)
        npd1 = norm.pdf(d1)
        
        # Delta
        if option_type.lower() == 'call':
            delta = nd1
        else:
            delta = nd1 - 1
        
        # Gamma (same for calls and puts)
        gamma = npd1 / (S * sigma * np.sqrt(T))
        
        # Theta
        theta_common = -(S * npd1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T)
        
        if option_type.lower() == 'call':
            theta = (theta_common * nd2) / 365  # Convert to daily theta
        else:
            theta = (theta_common * norm.cdf(-d2) + r * K * np.exp(-r * T)) / 365
        
        # Vega (same for calls and puts)
        vega = S * npd1 * np.sqrt(T) / 100  # Convert to 1% vol change
        
        # Rho
        if option_type.lower() == 'call':
            rho = K * T * np.exp(-r * T) * nd2 / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        greeks = {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
        
        logger.debug(f"Calculated Greeks: {greeks}")
        return greeks
        
    except Exception as e:
        logger.error(f"Error calculating Greeks: {str(e)}")
        return {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }

def calculate_implied_volatility(market_price: float, S: float, K: float, T: float, r: float, 
                               option_type: str = 'call', max_iterations: int = 100) -> Optional[float]:
    """
    Calculate implied volatility using Brent's method
    
    Args:
        market_price: Market price of the option
        S: Current price of underlying
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        option_type: 'call' or 'put'
        max_iterations: Maximum iterations for convergence
    
    Returns:
        Optional[float]: Implied volatility or None if cannot be calculated
    """
    try:
        if T <= 0 or market_price <= 0:
            return None
        
        # Intrinsic value
        if option_type.lower() == 'call':
            intrinsic = max(S - K, 0)
        else:
            intrinsic = max(K - S, 0)
        
        if market_price <= intrinsic:
            return None
        
        def price_difference(sigma):
            return black_scholes_price(S, K, T, r, sigma, option_type) - market_price
        
        # Use Brent's method to find root
        try:
            logger.debug(f"Attempting to find implied volatility using Brent's method")
            iv = brentq(price_difference, 0.001, 5.0, maxiter=max_iterations)
            logger.info(f"Found implied volatility: {iv:.4f}")
            return iv
        except ValueError:
            # If Brent's method fails, try a simple grid search
            best_iv = None
            min_diff = float('inf')
            
            for sigma in np.arange(0.01, 2.0, 0.01):
                price = black_scholes_price(S, K, T, r, sigma, option_type)
                diff = abs(price - market_price)
                
                if diff < min_diff:
                    min_diff = diff
                    best_iv = sigma
            
            if min_diff < market_price * 0.01:
                logger.info(f"Found implied volatility using grid search: {best_iv:.4f}")
                return best_iv
            else:
                logger.warning("Could not find acceptable implied volatility")
                return None
        
    except Exception as e:
        logger.error(f"Error calculating implied volatility: {str(e)}")
        return None

def binomial_option_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n: int = 100,
    option_type: str = 'call',
    exercise_type: str = 'european'
) -> float:
    """
    Calculate option price using binomial tree model
    
    Args:
        S: Current price of underlying
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility of underlying
        n: Number of time steps (default: 100)
        option_type: 'call' or 'put'
        exercise_type: 'european' or 'american'
    
    Returns:
        float: Option price
    """
    try:
        logger.debug(f"Starting binomial tree calculation with {n} steps")
        logger.debug(f"Parameters: S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
        
        # Calculate tree parameters
        dt: float = T / n
        u: float = np.exp(sigma * np.sqrt(dt))
        d: float = 1 / u
        p: float = (np.exp(r * dt) - d) / (u - d)
        
        logger.debug("Initializing binomial tree parameters")
        logger.debug(f"Time step: {dt:.6f}, up factor: {u:.4f}")
        logger.debug(f"Down factor: {d:.4f}, probability: {p:.4f}")
        
        # Initialize arrays with type hints
        prices: NDArray[np.float64] = np.zeros((n + 1, n + 1))
        option_values: NDArray[np.float64] = np.zeros((n + 1, n + 1))
        
        logger.debug("Building binomial tree price lattice")
        # Calculate underlying prices at each node
        for i in range(n + 1):
            for j in range(i + 1):
                prices[j, i] = S * (u ** (i - j)) * (d ** j)
        
        logger.debug("Calculating terminal option values")
        # Calculate option values at expiration
        for j in range(n + 1):
            if option_type.lower() == 'call':
                option_values[j, n] = max(0, prices[j, n] - K)
            else:
                option_values[j, n] = max(0, K - prices[j, n])
        
        logger.debug("Performing backward induction")
        # Backward induction
        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                # Calculate continuation value
                continuation: float = np.exp(-r * dt) * (p * option_values[j, i + 1] + 
                                                       (1 - p) * option_values[j + 1, i + 1])
                
                if exercise_type.lower() == 'american':
                    # Calculate exercise value
                    if option_type.lower() == 'call':
                        exercise: float = max(0, prices[j, i] - K)
                    else:
                        exercise: float = max(0, K - prices[j, i])
                    
                    option_values[j, i] = max(continuation, exercise)
                else:
                    option_values[j, i] = continuation
        
        price: float = option_values[0, 0]
        logger.info(f"Binomial tree calculation completed - Option price: {price:.4f}")
        
        return price
        
    except Exception as e:
        logger.error(f"Error in binomial option pricing: {str(e)}")
        return black_scholes_price(S, K, T, r, sigma, option_type)

def monte_carlo_option_price(
    S: float,
    K: Union[float, List[float]],
    T: float,
    r: float,
    sigma: float,
    n_simulations: int = 10000,
    option_type: str = 'call',
    batch_size: int = 1000
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate option price using Monte Carlo simulation with batch processing
    
    Args:
        S: Current price of underlying
        K: Strike price or list of strike prices
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility of underlying
        n_simulations: Number of Monte Carlo simulations (default: 10000)
        option_type: 'call' or 'put'
        batch_size: Number of simulations per batch (default: 1000)
    
    Returns:
        Union[float, List[float]]: Option price(s)
    """
    try:
        # Handle single strike price case
        single_strike = not isinstance(K, list)
        strikes = [K] if single_strike else K
        
        logger.debug(f"Starting Monte Carlo simulation with {n_simulations} paths")
        logger.debug(f"Parameters: S={S}, strikes={strikes}, T={T}, r={r}, sigma={sigma}")
        logger.debug(f"Using batch size of {batch_size}")
        
        np.random.seed(42)  # For reproducible results
        
        results = []
        n_batches = n_simulations // batch_size
        
        for strike in strikes:
            total_payoffs: NDArray[np.float64] = np.zeros(n_simulations)
            
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                
                # Generate random price paths for this batch
                Z: NDArray[np.float64] = np.random.standard_normal(batch_size)
                ST: NDArray[np.float64] = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
                
                # Calculate payoffs for this batch
                if option_type.lower() == 'call':
                    payoffs: NDArray[np.float64] = np.maximum(ST - strike, 0)
                else:
                    payoffs: NDArray[np.float64] = np.maximum(strike - ST, 0)
                
                total_payoffs[start_idx:end_idx] = payoffs
                
                if (batch + 1) % 10 == 0:
                    logger.debug(f"Completed {batch + 1}/{n_batches} batches for strike {strike}")
            
            # Calculate statistics
            mean_payoff: float = np.mean(total_payoffs)
            std_payoff: float = np.std(total_payoffs)
            
            # Discount to present value
            option_price: float = np.exp(-r * T) * mean_payoff
            
            # Calculate confidence interval
            confidence_level: float = 0.95
            z_score: float = norm.ppf((1 + confidence_level) / 2)
            margin_of_error: float = z_score * std_payoff / np.sqrt(n_simulations)
            ci_lower: float = option_price - margin_of_error
            ci_upper: float = option_price + margin_of_error
            
            logger.debug(f"Strike {strike}:")
            logger.debug(f"  Option price: {option_price:.4f}")
            logger.debug(f"  {confidence_level*100}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            logger.debug(f"  Standard Error: {margin_of_error:.4f}")
            
            results.append(option_price)
        
        logger.info("Monte Carlo simulation completed successfully")
        return results[0] if single_strike else results
        
    except Exception as e:
        logger.error(f"Error in Monte Carlo option pricing: {str(e)}")
        return black_scholes_price(S, K, T, r, sigma, option_type)

def calculate_option_strategy_payoff(
    strategy_legs: List[Dict[str, Union[str, float, int]]],
    underlying_prices: NDArray[np.float64]
) -> Optional[Dict[str, Union[NDArray[np.float64], float, List[float]]]]:
    """
    Calculate payoff for an options strategy
    
    Args:
        strategy_legs (list): List of strategy legs, each containing:
            - option_type: 'call' or 'put'
            - strike: Strike price
            - quantity: Number of contracts (positive for long, negative for short)
            - premium: Premium paid/received
        underlying_prices (np.array): Array of underlying prices for payoff calculation
    
    Returns:
        dict: Strategy payoff analysis
    """
    try:
        logger.debug(f"Calculating strategy payoff for {len(strategy_legs)} legs")
        logger.debug(f"Price range: {underlying_prices[0]} to {underlying_prices[-1]}")
        
        payoffs = np.zeros_like(underlying_prices)
        total_premium = 0
        
        logger.info("Processing strategy legs:")
        
        for leg in strategy_legs:
            option_type = leg['option_type'].lower()
            strike = leg['strike']
            quantity = leg['quantity']
            premium = leg['premium']
            
            # Calculate intrinsic values
            if option_type == 'call':
                intrinsic_values = np.maximum(underlying_prices - strike, 0)
            else:
                intrinsic_values = np.maximum(strike - underlying_prices, 0)
            
            logger.debug(f"Processing {option_type} option: Strike={strike}, Quantity={quantity}")
            
            # Add to total payoff
            payoffs += quantity * intrinsic_values
            total_premium += quantity * premium
            
            logger.debug(f"Leg contribution - Premium: {quantity * premium}, Max Payoff: {np.max(quantity * intrinsic_values)}")
        
        # Calculate net payoff and metrics
        logger.debug("Calculating final strategy metrics")
        net_payoffs = payoffs - total_premium
        
        # Calculate key metrics
        max_profit = np.max(net_payoffs)
        max_loss = np.min(net_payoffs)
        breakeven_points = []
        
        # Find breakeven points (where net payoff crosses zero)
        logger.debug("Finding breakeven points")
        for i in range(len(net_payoffs) - 1):
            if (net_payoffs[i] <= 0 and net_payoffs[i + 1] >= 0) or \
               (net_payoffs[i] >= 0 and net_payoffs[i + 1] <= 0):
                # Linear interpolation to find exact breakeven
                x1, x2 = underlying_prices[i], underlying_prices[i + 1]
                y1, y2 = net_payoffs[i], net_payoffs[i + 1]
                breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
                breakeven_points.append(breakeven)
                logger.debug(f"Found breakeven point at {breakeven:.2f}")
        
        result = {
            'underlying_prices': underlying_prices,
            'payoffs': payoffs,
            'net_payoffs': net_payoffs,
            'total_premium': total_premium,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven_points': breakeven_points
        }
        
        logger.info(f"Strategy analysis complete - Max Profit: {max_profit:.2f}, Max Loss: {max_loss:.2f}")
        logger.debug(f"Found {len(breakeven_points)} breakeven points: {[f'{x:.2f}' for x in breakeven_points]}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating strategy payoff: {str(e)}")
        return None

def get_options_chain_data(
    symbol: str,
    current_price: float,
    expiry_dates: List[Union[str, datetime]],
    risk_free_rate: float = 0.065,
    volatility: float = 0.25
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Generate synthetic options chain data for a commodity
    
    Args:
        symbol: Commodity symbol
        current_price: Current underlying price
        expiry_dates: List of expiry dates (str in YYYY-MM-DD format or datetime objects)
        risk_free_rate: Risk-free interest rate (default: 0.065)
        volatility: Assumed volatility (default: 0.25)
    
    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: Options chain data for each expiry date
    """
    try:
        logger.info(f"Generating options chain data for {symbol} at price {current_price}")
        logger.debug(f"Parameters - Risk-free rate: {risk_free_rate}, Volatility: {volatility}")
        
        options_data = {}
        current_datetime = datetime.now()
        
        for expiry_date in expiry_dates:
            # Calculate time to expiry
            if isinstance(expiry_date, str):
                expiry_dt = datetime.strptime(expiry_date, '%Y-%m-%d')
            else:
                expiry_dt = expiry_date
            
            time_to_expiry = max((expiry_dt - current_datetime).days / 365.0, 0.001)
            logger.debug(f"Processing expiry {expiry_dt.strftime('%Y-%m-%d')} - TTM: {time_to_expiry:.4f} years")
            
            # Generate strike prices around current price
            atm_strike = round(current_price / 50) * 50
            strikes = [atm_strike + (i * 50) for i in range(-10, 11)]
            logger.debug(f"Generated {len(strikes)} strikes from {strikes[0]} to {strikes[-1]}")
            
            calls_data = []
            puts_data = []
            
            logger.debug("Calculating option prices and Greeks for each strike")
            for strike in strikes:
                logger.debug(f"Processing strike {strike}")
                # Calculate option prices and Greeks
                call_price = black_scholes_price(current_price, strike, time_to_expiry, risk_free_rate, volatility, 'call')
                put_price = black_scholes_price(current_price, strike, time_to_expiry, risk_free_rate, volatility, 'put')
                
                call_greeks = calculate_greeks(current_price, strike, time_to_expiry, risk_free_rate, volatility, 'call')
                put_greeks = calculate_greeks(current_price, strike, time_to_expiry, risk_free_rate, volatility, 'put')
                
                # Add synthetic bid-ask spreads and volume/OI
                call_bid = call_price * 0.98
                call_ask = call_price * 1.02
                put_bid = put_price * 0.98
                put_ask = put_price * 1.02
                
                # Simulate volume and open interest based on moneyness
                moneyness = abs(current_price - strike) / current_price
                volume_factor = max(0.1, 1 - moneyness * 2)  # Higher volume for ATM options
                
                base_volume = int(np.random.uniform(1000, 20000) * volume_factor)
                base_oi = int(np.random.uniform(500, 10000) * volume_factor)
                
                calls_data.append({
                    'strike': strike,
                    'lastPrice': call_price,
                    'bid': call_bid,
                    'ask': call_ask,
                    'volume': base_volume,
                    'openInterest': base_oi,
                    'impliedVolatility': volatility,
                    'delta': call_greeks['delta'],
                    'gamma': call_greeks['gamma'],
                    'theta': call_greeks['theta'],
                    'vega': call_greeks['vega']
                })
                
                puts_data.append({
                    'strike': strike,
                    'lastPrice': put_price,
                    'bid': put_bid,
                    'ask': put_ask,
                    'volume': base_volume,
                    'openInterest': base_oi,
                    'impliedVolatility': volatility,
                    'delta': put_greeks['delta'],
                    'gamma': put_greeks['gamma'],
                    'theta': put_greeks['theta'],
                    'vega': put_greeks['vega']
                })
            
            expiry_key = expiry_dt.strftime('%Y-%m-%d')
            options_data[expiry_key] = {
                'calls': pd.DataFrame(calls_data),
                'puts': pd.DataFrame(puts_data)
            }
            logger.info(f"Completed calculations for expiry {expiry_key}")
        
        logger.info(f"Successfully generated options chain data for {len(expiry_dates)} expiry dates")
        return options_data
        
    except Exception as e:
        logger.error(f"Error generating options chain data: {str(e)}")
        return {}

def calculate_portfolio_greeks(positions):
    """
    Calculate portfolio-level Greeks from individual positions
    
    Args:
        positions (list): List of option positions with Greeks
    
    Returns:
        dict: Portfolio Greeks
    """
    try:
        portfolio_greeks = {
            'delta': 0,
            'gamma': 0,
            'theta': 0,
            'vega': 0,
            'rho': 0
        }
        
        for position in positions:
            quantity = position.get('quantity', 1)
            position_greeks = position.get('greeks', {})
            
            # Adjust for position direction (buy = positive, sell = negative)
            multiplier = quantity if position.get('side', 'buy').lower() == 'buy' else -quantity
            
            for greek in portfolio_greeks:
                portfolio_greeks[greek] += position_greeks.get(greek, 0) * multiplier
        
        return portfolio_greeks
        
    except Exception as e:
        logger.error(f"Error calculating portfolio Greeks: {str(e)}")
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

def analyze_energy_commodities(
    symbol: str,
    current_price: float,
    historical_prices: NDArray[np.float64],
    options_chain: Dict[str, pd.DataFrame],
    risk_tolerance: str = 'moderate'  # 'conservative', 'moderate', 'aggressive'
) -> Dict[str, Any]:
    """
    Analyze energy commodity options (Crude Oil and Natural Gas) and provide trading recommendations
    
    Args:
        symbol: Commodity symbol (e.g., 'CRUDEOIL', 'NATURALGAS')
        current_price: Current price of the underlying
        historical_prices: Historical price data (last 30 days)
        options_chain: Current options chain data
        risk_tolerance: Investor's risk tolerance level
    
    Returns:
        Dict containing trading recommendations and analysis with commodity-specific insights
    """
    try:
        logger.info(f"Analyzing {symbol} options for trading opportunities")
        
        # Analyze trading range characteristics
        range_analysis = analyze_trading_range(historical_prices)
        logger.info(f"Range Analysis - Width: {range_analysis['range_width']}%, Adherence: {range_analysis['range_adherence']}%")
        
        # Calculate technical indicators
        sma_20: float = np.mean(historical_prices[-20:])
        sma_50: float = np.mean(historical_prices[-50:])
        rsi: float = calculate_rsi(historical_prices)
        
        # Calculate volatility
        returns: NDArray[np.float64] = np.log(historical_prices[1:] / historical_prices[:-1])
        hist_vol: float = np.std(returns) * np.sqrt(252)
        
        # Determine market characteristics
        trend: str = "Range-Bound" if range_analysis['range_adherence'] > 70 else "Bullish" if sma_20 > sma_50 else "Bearish"
        momentum: str = "Strong" if abs(sma_20 - sma_50) / sma_50 > 0.02 else "Weak"
        
        logger.info(f"Starting analysis for {symbol}")
        logger.info(f"Market Structure: {trend} with {momentum} momentum")
        logger.debug(f"Technical Indicators - SMA20: {sma_20:.2f}, SMA50: {sma_50:.2f}, RSI: {rsi:.2f}")
        logger.debug(f"Volatility: {hist_vol:.2%}, Range Width: {range_analysis['range_width']:.2f}%")
        
        # Support and resistance levels
        support = round(min(historical_prices[-20:]) / (50 if symbol == 'CRUDEOIL' else 10)) * (50 if symbol == 'CRUDEOIL' else 10)
        resistance = round(max(historical_prices[-20:]) / (50 if symbol == 'CRUDEOIL' else 10)) * (50 if symbol == 'CRUDEOIL' else 10)
        
        # Commodity-specific analysis
        if symbol == 'CRUDEOIL':
            logger.info("Analyzing Crude Oil specific patterns")
            commodity_factors = {
                'inventory_impact': "high" if hist_vol > 0.35 else "moderate",
                'opec_impact': True if hist_vol > 0.40 else False,
                'seasonality': 'Summer driving season' if datetime.now().month in [5,6,7,8] else 'Off-peak',
                'key_levels': {
                    'support': support,
                    'resistance': resistance
                }
            }
        else:  # NATURALGAS
            logger.info("Analyzing Natural Gas specific patterns")
            current_month = datetime.now().month
            commodity_factors = {
                'weather_impact': "high" if current_month in [1,2,7,8] else "moderate",
                'storage_impact': "high" if hist_vol > 0.45 else "moderate",
                'seasonality': 'Peak demand' if current_month in [1,2,7,8] else 'Off-peak',
                'key_levels': {
                    'support': support,
                    'resistance': resistance
                }
            }

        # Technical analysis
        sma_20: float = np.mean(historical_prices[-20:])
        sma_50: float = np.mean(historical_prices[-50:])
        rsi: float = calculate_rsi(historical_prices)
        
        # Volatility analysis with commodity-specific thresholds
        returns: NDArray[np.float64] = np.log(historical_prices[1:] / historical_prices[:-1])
        hist_vol: float = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Commodity-specific volatility thresholds
        vol_threshold = 0.35 if symbol == 'CRUDEOIL' else 0.45  # Natural gas typically more volatile
        
        # Market trend analysis with commodity-specific momentum thresholds
        trend: str = "Bullish" if sma_20 > sma_50 else "Bearish"
        momentum_threshold = 0.02 if symbol == 'CRUDEOIL' else 0.03
        momentum: str = "Strong" if abs(sma_20 - sma_50) / sma_50 > momentum_threshold else "Weak"
        
        logger.debug(f"Market Analysis - Trend: {trend}, Momentum: {momentum}")
        logger.debug(f"Historical Volatility: {hist_vol:.2%}")
        
        # Find ATM options
        atm_strike = round(current_price / 50) * 50
        
        recommendations = []
        if trend == "Bullish":
            if risk_tolerance == 'conservative':
                recommendations.append({
                    'strategy': 'Bull Call Spread',
                    'actions': [
                        f'Buy {symbol} {atm_strike} Call',
                        f'Sell {symbol} {atm_strike + 100} Call'
                    ],
                    'rationale': 'Limited risk bullish strategy with defined profit potential',
                    'risk_level': 'Low',
                    'target_return': '15-20%'
                })
            elif risk_tolerance == 'moderate':
                recommendations.append({
                    'strategy': 'Long Call',
                    'actions': [f'Buy {symbol} {atm_strike} Call'],
                    'rationale': 'Direct bullish exposure with defined risk',
                    'risk_level': 'Medium',
                    'target_return': '25-40%'
                })
            else:  # aggressive
                recommendations.append({
                    'strategy': 'Call Ratio Backspread',
                    'actions': [
                        f'Sell 1 {symbol} {atm_strike} Call',
                        f'Buy 2 {symbol} {atm_strike + 50} Calls'
                    ],
                    'rationale': 'Leveraged bullish strategy for strong upward moves',
                    'risk_level': 'High',
                    'target_return': '50%+'
                })
        else:  # Bearish
            if risk_tolerance == 'conservative':
                recommendations.append({
                    'strategy': 'Bear Put Spread',
                    'actions': [
                        f'Buy {symbol} {atm_strike} Put',
                        f'Sell {symbol} {atm_strike - 100} Put'
                    ],
                    'rationale': 'Limited risk bearish strategy with defined profit potential',
                    'risk_level': 'Low',
                    'target_return': '15-20%'
                })
            elif risk_tolerance == 'moderate':
                recommendations.append({
                    'strategy': 'Long Put',
                    'actions': [f'Buy {symbol} {atm_strike} Put'],
                    'rationale': 'Direct bearish exposure with defined risk',
                    'risk_level': 'Medium',
                    'target_return': '25-40%'
                })
            else:  # aggressive
                recommendations.append({
                    'strategy': 'Put Ratio Backspread',
                    'actions': [
                        f'Sell 1 {symbol} {atm_strike} Put',
                        f'Buy 2 {symbol} {atm_strike - 50} Puts'
                    ],
                    'rationale': 'Leveraged bearish strategy for strong downward moves',
                    'risk_level': 'High',
                    'target_return': '50%+'
                })
        
        # Get upcoming events and their impact
        events = get_upcoming_events(symbol)
        event_risk = 'High' if any(e['impact'] == 'High' for e in events) else 'Moderate'
        logger.info(f"Event risk assessment: {event_risk}")
        
        # Commodity-specific volatility thresholds
        vol_threshold = 0.35 if symbol == 'CRUDEOIL' else 0.45
        
        # Get range-bound trading strategies if applicable
        range_strategies = []
        if trend == "Range-Bound":
            range_strategies = get_range_trading_strategies(
                current_price=current_price,
                range_analysis=range_analysis,
                hist_vol=hist_vol,
                symbol=symbol
            )
            logger.info(f"Generated {len(range_strategies)} range-bound trading strategies")
        
        # Get directional trading suggestions for trending markets
        directional_strategies = get_trading_suggestions(
            symbol=symbol,
            trend=trend,
            rsi=rsi,
            hist_vol=hist_vol,
            current_price=current_price,
            support_level=commodity_factors['key_levels']['support'],
            resistance_level=commodity_factors['key_levels']['resistance']
        )
        logger.info(f"Generated {len(directional_strategies)} directional trading strategies")
        
        # Combine strategies based on market conditions
        if trend == "Range-Bound":
            primary_strategies = range_strategies
            secondary_strategies = directional_strategies
        else:
            primary_strategies = directional_strategies
            secondary_strategies = range_strategies
        
        # Organize analysis into comprehensive result with range focus
        result = {
            'symbol': symbol,
            'current_price': current_price,
            'summary': {
                'market_structure': trend,
                'overall_bias': f"{trend} with {momentum} momentum",
                'range_analysis': {
                    'width': f"{range_analysis['range_width']:.1f}%",
                    'adherence': f"{range_analysis['range_adherence']:.1f}%",
                    'boundaries': f"High: {range_analysis['range_high']}, Low: {range_analysis['range_low']}"
                },
                'volatility_regime': f"{'High' if hist_vol > vol_threshold else 'Normal'} volatility environment",
                'next_event': events[0] if events else 'No major events upcoming',
                'primary_strategy': primary_strategies[0]['strategy'] if primary_strategies else 'No clear setup'
            },
            'market_conditions': {
                'range_characteristics': {
                    'state': 'Established' if range_analysis['range_adherence'] > 70 else 'Developing',
                    'position': get_range_position(current_price, range_analysis),
                    'boundaries': {
                        'high': range_analysis['range_high'],
                        'mid': range_analysis['range_mid'],
                        'low': range_analysis['range_low']
                    },
                    'width_analysis': {
                        'percentage': range_analysis['range_width'],
                        'interpretation': 'Wide' if range_analysis['range_width'] > 15 else 'Narrow'
                    }
                },
                'trend': {
                    'direction': trend,
                    'momentum': momentum,
                    'strength': 'Strong' if abs(sma_20 - sma_50) / sma_50 > 0.02 else 'Weak',
                    'moving_averages': {
                        'sma_20': round(sma_20, 2),
                        'sma_50': round(sma_50, 2)
                    }
                },
                'volatility': {
                    'historical': f"{hist_vol:.2%}",
                    'regime': 'High' if hist_vol > vol_threshold else 'Normal',
                    'range_volatility': f"{range_analysis['range_width'] / np.sqrt(30):.1f}% daily",
                    'implication': get_volatility_implication(hist_vol, range_analysis['range_width'])
                },
                'technical': {
                    'rsi': round(rsi, 2),
                    'rsi_condition': 'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral',
                    'key_levels': commodity_factors['key_levels']
                }
            },
            'commodity_factors': commodity_factors,
            'event_risk': {
                'upcoming_events': events,
                'risk_level': event_risk,
                'trading_window': 'Pre-event' if any(
                    datetime.strptime(e['date'], '%Y-%m-%d').date() - datetime.now().date() <= timedelta(days=2)
                    for e in events
                ) else 'Normal'
            },
            'trading_plan': {
                'primary_strategies': primary_strategies,
                'secondary_strategies': secondary_strategies,
                'risk_management': {
                    'position_size': get_position_size_recommendation(hist_vol, range_analysis['range_adherence']),
                    'stop_placement': {
                        'range_stops': {
                            'long': f"Below {round(range_analysis['range_low'] * 0.95, 2)}",
                            'short': f"Above {round(range_analysis['range_high'] * 1.05, 2)}"
                        },
                        'volatility_based': f"{round(hist_vol * current_price * 0.1, 2)} points from entry"
                    },
                    'profit_targets': {
                        'range_based': f"70% of range width ({round(range_analysis['range_width'] * 0.7, 1)}%)",
                        'time_based': '50% of option premium in 50% of time to expiry'
                    },
                    'adjustments': {
                        'volatility': 'Reduce position size by 50%' if hist_vol > vol_threshold else 'Standard sizing',
                        'event_risk': 'Consider straddle/strangle' if event_risk == 'High' else 'Standard strategies',
                        'range_breach': 'Convert to directional strategy on valid range break'
                    }
                }
            }
        }
        
        # Log analysis results
        logger.info(f"Analysis completed for {symbol}")
        logger.info(f"Market Bias: {result['summary']['overall_bias']}")
        logger.info(f"Volatility Regime: {result['market_conditions']['volatility']['regime']}")
        logger.info(f"Event Risk Level: {result['event_risk']['risk_level']}")
        logger.info(f"Generated {len(trading_suggestions)} trading suggestions")
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing options: {str(e)}")
        return {
            'symbol': symbol,
            'error': 'Unable to generate recommendations',
            'message': str(e)
        }

def calculate_rsi(prices: NDArray[np.float64], period: int = 14) -> float:
    """
    Calculate the Relative Strength Index (RSI) technical indicator
    
    Args:
        prices: Array of historical prices
        period: RSI period (default: 14)
    
    Returns:
        float: RSI value between 0 and 100
    """
    try:
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi, 2)
        
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        return 50.0  # Return neutral RSI on error

def get_upcoming_events(symbol: str) -> List[Dict[str, str]]:
    """
    Get upcoming events that could impact commodity prices
    
    Args:
        symbol: Commodity symbol ('CRUDEOIL' or 'NATURALGAS')
    
    Returns:
        List of upcoming events with dates and descriptions
    """
    try:
        current_date = datetime.now()
        events = []
        
        if symbol == 'CRUDEOIL':
            # Weekly EIA Petroleum Status Report (every Wednesday)
            next_wednesday = current_date + timedelta(days=(2 - current_date.weekday() + 7) % 7)
            events.append({
                'date': next_wednesday.strftime('%Y-%m-%d'),
                'event': 'EIA Petroleum Status Report',
                'impact': 'High'
            })
            
            # Monthly OPEC Meeting
            next_month = current_date.replace(day=1) + timedelta(days=32)
            opec_meeting = next_month.replace(day=5)  # Typically early in the month
            events.append({
                'date': opec_meeting.strftime('%Y-%m-%d'),
                'event': 'OPEC+ Monthly Meeting',
                'impact': 'High'
            })
            
        else:  # NATURALGAS
            # Weekly Natural Gas Storage Report (every Thursday)
            next_thursday = current_date + timedelta(days=(3 - current_date.weekday() + 7) % 7)
            events.append({
                'date': next_thursday.strftime('%Y-%m-%d'),
                'event': 'EIA Natural Gas Storage Report',
                'impact': 'High'
            })
            
            # Weather forecast updates (twice weekly)
            next_monday = current_date + timedelta(days=(0 - current_date.weekday() + 7) % 7)
            events.append({
                'date': next_monday.strftime('%Y-%m-%d'),
                'event': 'NOAA 6-10 Day Weather Forecast',
                'impact': 'Medium'
            })
        
        return events
        
    except Exception as e:
        logger.error(f"Error getting upcoming events: {str(e)}")
        return []

def analyze_trading_range(
    historical_prices: NDArray[np.float64],
    lookback_period: int = 30
) -> Dict[str, float]:
    """
    Analyze trading range characteristics
    
    Args:
        historical_prices: Array of historical prices
        lookback_period: Period for range analysis (default: 30 days)
    
    Returns:
        Dict containing range analysis metrics
    """
    try:
        recent_prices = historical_prices[-lookback_period:]
        range_high = np.max(recent_prices)
        range_low = np.min(recent_prices)
        range_mid = (range_high + range_low) / 2
        range_width = (range_high - range_low) / range_mid * 100  # as percentage
        
        # Calculate how often price stays within range
        within_range = np.sum((recent_prices >= range_low) & (recent_prices <= range_high))
        range_adherence = within_range / len(recent_prices) * 100
        
        return {
            'range_high': round(range_high, 2),
            'range_low': round(range_low, 2),
            'range_mid': round(range_mid, 2),
            'range_width': round(range_width, 2),
            'range_adherence': round(range_adherence, 2)
        }
    except Exception as e:
        logger.error(f"Error analyzing trading range: {str(e)}")
        return {}

def get_range_trading_strategies(
    current_price: float,
    range_analysis: Dict[str, float],
    hist_vol: float,
    symbol: str
) -> List[Dict[str, Any]]:
    """
    Generate range-bound trading strategies
    
    Args:
        current_price: Current price of the underlying
        range_analysis: Results from range analysis
        hist_vol: Historical volatility
        symbol: Commodity symbol
    
    Returns:
        List of range-trading strategies
    """
    try:
        strategies = []
        range_high = range_analysis['range_high']
        range_low = range_analysis['range_low']
        range_mid = range_analysis['range_mid']
        
        # Distance from current price to range boundaries
        dist_to_high = (range_high - current_price) / current_price * 100
        dist_to_low = (current_price - range_low) / current_price * 100
        
        # Near bottom of range
        if dist_to_low < 5:
            strategies.append({
                'strategy': 'Range Bottom Iron Butterfly',
                'setup': {
                    'primary': f'Sell ATM {symbol} Put at {round(current_price, 2)}',
                    'wings': [
                        f'Buy {symbol} Put at {round(range_low * 0.95, 2)}',
                        f'Buy {symbol} Call at {round(range_mid, 2)}'
                    ]
                },
                'rationale': 'High probability of price bounce from range bottom',
                'profit_zone': f'Between {round(range_low, 2)} and {round(range_mid, 2)}',
                'max_profit': 'Net premium received',
                'stop_level': f'Below {round(range_low * 0.95, 2)}'
            })
        
        # Near top of range
        elif dist_to_high < 5:
            strategies.append({
                'strategy': 'Range Top Iron Butterfly',
                'setup': {
                    'primary': f'Sell ATM {symbol} Call at {round(current_price, 2)}',
                    'wings': [
                        f'Buy {symbol} Call at {round(range_high * 1.05, 2)}',
                        f'Buy {symbol} Put at {round(range_mid, 2)}'
                    ]
                },
                'rationale': 'High probability of price rejection from range top',
                'profit_zone': f'Between {round(range_mid, 2)} and {round(range_high, 2)}',
                'max_profit': 'Net premium received',
                'stop_level': f'Above {round(range_high * 1.05, 2)}'
            })
        
        # Middle of range
        else:
            strategies.append({
                'strategy': 'Iron Condor',
                'setup': {
                    'sells': [
                        f'Sell {symbol} Put at {round(range_low * 1.05, 2)}',
                        f'Sell {symbol} Call at {round(range_high * 0.95, 2)}'
                    ],
                    'buys': [
                        f'Buy {symbol} Put at {round(range_low * 0.95, 2)}',
                        f'Buy {symbol} Call at {round(range_high * 1.05, 2)}'
                    ]
                },
                'rationale': 'Profit from time decay while price remains in range',
                'profit_zone': f'Between {round(range_low * 1.05, 2)} and {round(range_high * 0.95, 2)}',
                'max_profit': 'Net premium received',
                'adjustment': 'Roll positions if price approaches range boundaries'
            })
        
        # Add mean reversion strategy if volatility is high
        if hist_vol > 0.30:
            strategies.append({
                'strategy': 'Mean Reversion Calendar Spread',
                'setup': {
                    'front_month': f'Sell {symbol} Straddle at {round(current_price, 2)}',
                    'back_month': f'Buy {symbol} Strangle at {round(range_low, 2)}/{round(range_high, 2)}'
                },
                'rationale': 'Profit from volatility crush and mean reversion',
                'profit_zone': 'Maximum at mean reversion point',
                'adjustment': 'Adjust back month strikes if range shifts'
            })
        
        return strategies
    except Exception as e:
        logger.error(f"Error generating range trading strategies: {str(e)}")
        return []

def get_range_position(current_price: float, range_analysis: Dict[str, float]) -> str:
    """
    Determine the current price position within the trading range
    
    Args:
        current_price: Current price of the underlying
        range_analysis: Results from range analysis
    
    Returns:
        str: Description of price position within range
    """
    try:
        range_high = range_analysis['range_high']
        range_low = range_analysis['range_low']
        range_mid = range_analysis['range_mid']
        
        # Calculate percentage distance from boundaries
        dist_from_high = (range_high - current_price) / (range_high - range_low) * 100
        dist_from_low = (current_price - range_low) / (range_high - range_low) * 100
        
        if dist_from_high <= 10:
            return "Near Range Top"
        elif dist_from_low <= 10:
            return "Near Range Bottom"
        elif abs(current_price - range_mid) / (range_high - range_low) * 100 <= 10:
            return "Mid-Range"
        elif current_price > range_mid:
            return "Upper Half of Range"
        else:
            return "Lower Half of Range"
            
    except Exception as e:
        logger.error(f"Error determining range position: {str(e)}")
        return "Unknown"

def get_volatility_implication(hist_vol: float, range_width: float) -> str:
    """
    Get trading implications based on volatility and range characteristics
    
    Args:
        hist_vol: Historical volatility
        range_width: Width of trading range as percentage
    
    Returns:
        str: Trading implication based on volatility analysis
    """
    try:
        # Convert range width to annualized volatility equivalent
        range_vol = range_width / np.sqrt(30) * np.sqrt(252)
        
        if hist_vol > range_vol * 1.5:
            return "Range likely to expand - wider stops needed"
        elif hist_vol < range_vol * 0.5:
            return "Range likely to contract - tighter stops possible"
        elif hist_vol > 0.35:
            return "High volatility - reduce position sizes"
        else:
            return "Normal volatility - standard position sizing"
            
    except Exception as e:
        logger.error(f"Error analyzing volatility implications: {str(e)}")
        return "Use standard position sizing"

def get_position_size_recommendation(hist_vol: float, range_adherence: float) -> str:
    """
    Get position size recommendation based on volatility and range characteristics
    
    Args:
        hist_vol: Historical volatility
        range_adherence: How well price stays within range
    
    Returns:
        str: Position size recommendation
    """
    try:
        base_size = "2-3% of portfolio per trade"
        
        if hist_vol > 0.35:
            if range_adherence > 70:
                return f"1-2% of portfolio (High volatility but strong range)"
            else:
                return f"1% of portfolio (High volatility, weak range)"
        else:
            if range_adherence > 70:
                return f"{base_size} (Strong range provides good risk/reward)"
            else:
                return f"1.5-2% of portfolio (Developing range requires caution)"
                
    except Exception as e:
        logger.error(f"Error determining position size: {str(e)}")
        return "2% of portfolio per trade"

def get_trading_suggestions(
    symbol: str,
    trend: str,
    rsi: float,
    hist_vol: float,
    current_price: float,
    support_level: float,
    resistance_level: float,
    range_analysis: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Get specific trading suggestions based on technical analysis and commodity type
    
    Args:
        symbol: Commodity symbol ('CRUDEOIL' or 'NATURALGAS')
        trend: Current market trend ('Bullish', 'Bearish', or 'Range-Bound')
        rsi: Current RSI value
        hist_vol: Historical volatility
        current_price: Current price of the underlying
        support_level: Nearest support level
        resistance_level: Nearest resistance level
        range_analysis: Optional range analysis results
    
    Returns:
        List of specific trading suggestions with entry, exit, and risk management
    """
    try:
        logger.debug(f"Generating trading suggestions for {symbol} in {trend} market")
        suggestions = []
        
        # Determine market conditions
        high_vol_threshold = 0.35 if symbol == 'CRUDEOIL' else 0.45
        is_high_vol = hist_vol > high_vol_threshold
        
        # Get range characteristics if available
        if range_analysis:
            range_position = get_range_position(current_price, range_analysis)
            range_width = range_analysis['range_width']
            range_adherence = range_analysis['range_adherence']
            logger.debug(f"Range Analysis - Position: {range_position}, Width: {range_width}%, Adherence: {range_adherence}%")
        
        # Calculate distances to key levels
        if range_analysis:
            dist_to_high = (range_analysis['range_high'] - current_price) / current_price
            dist_to_low = (current_price - range_analysis['range_low']) / current_price
            dist_to_mid = abs(current_price - range_analysis['range_mid']) / current_price
        else:
            dist_to_high = (resistance_level - current_price) / current_price
            dist_to_low = (current_price - support_level) / current_price
            dist_to_mid = 0

        logger.debug(f"Price distances - To High: {dist_to_high:.1%}, To Low: {dist_to_low:.1%}")

        if trend == 'Range-Bound' and range_analysis:
            # Range-bound strategies
            if range_position == "Near Range Top":
                suggestion = {
                    'strategy': 'Range Top Fade',
                    'setup': {
                        'primary': f'Sell {symbol} Call Spread',
                        'strikes': [
                            f'Sell {round(current_price/50)*50} Call',
                            f'Buy {round(range_analysis["range_high"]*1.05/50)*50} Call'
                        ]
                    },
                    'rationale': 'High probability of price rejection at range top',
                    'stop_level': f'Above {round(range_analysis["range_high"]*1.05, 2)}',
                    'target': f'50% of range width (${round(range_width/2, 2)})'
                }
                suggestions.append(suggestion)

            elif range_position == "Near Range Bottom":
                suggestion = {
                    'strategy': 'Range Bottom Bounce',
                    'setup': {
                        'primary': f'Buy {symbol} Call Spread',
                        'strikes': [
                            f'Buy {round(current_price/50)*50} Call',
                            f'Sell {round(range_analysis["range_mid"]/50)*50} Call'
                        ]
                    },
                    'rationale': 'High probability of price bounce from range bottom',
                    'stop_level': f'Below {round(range_analysis["range_low"]*0.95, 2)}',
                    'target': f'Range midpoint at {round(range_analysis["range_mid"], 2)}'
                }
                suggestions.append(suggestion)

            # Add Iron Condor for high volatility range conditions
            if is_high_vol and range_adherence > 70:
                suggestion = {
                    'strategy': 'High Vol Iron Condor',
                    'setup': {
                        'wings': [
                            f'Sell {round(current_price*0.95/50)*50} Put',
                            f'Buy {round(current_price*0.90/50)*50} Put',
                            f'Sell {round(current_price*1.05/50)*50} Call',
                            f'Buy {round(current_price*1.10/50)*50} Call'
                        ]
                    },
                    'rationale': 'Collect premium while price remains in established range',
                    'max_profit': 'Net premium received',
                    'adjustment': 'Roll positions if price approaches short strikes'
                }
                suggestions.append(suggestion)

        elif trend == 'Bullish':
            if rsi < 70:  # Not overbought
                if dist_to_high > 0.03:  # Room to run
                    suggestion = {
                        'strategy': 'Momentum Call Strategy',
                        'setup': {
                            'primary': f'Buy {symbol} Call Option',
                            'strike': round(current_price / 50) * 50,
                            'expiry': '1-2 months out',
                            'entry_levels': {
                                'ideal': round(current_price, 2),
                                'max': round(current_price * 1.01, 2)
                            }
                        },
                        'exit_plan': {
                            'target': round(current_price * (1 + dist_to_high), 2),
                            'stop_loss': round(current_price * (1 - dist_to_low * 0.5), 2),
                            'time_stop': '50% of time to expiry'
                        },
                        'position_sizing': {
                            'size': '2% of portfolio',
                            'adjustment': 'Reduce by 50%' if is_high_vol else 'Standard size'
                        },
                        'rationale': [
                            'Bullish trend confirmation',
                            f"RSI at {rsi:.1f} shows momentum potential",
                            f"{(dist_to_high*100):.1f}% room to range high/resistance"
                        ]
                    }
                    suggestions.append(suggestion)
            
            if is_high_vol:
                # High volatility premium collection strategy
                suggestion = {
                    'strategy': 'High Vol Put Write',
                    'setup': {
                        'primary': f'Sell {symbol} Put Option',
                        'strike': round((current_price * (1 - dist_to_low * 0.5)) / 50) * 50,
                        'expiry': '2-3 weeks out',
                        'entry_levels': {
                            'ideal': round(current_price, 2),
                            'min': round(current_price * 0.98, 2)
                        }
                    },
                    'exit_plan': {
                        'target': '50% of premium collected',
                        'stop_loss': round(current_price * (1 - dist_to_low * 0.7), 2),
                        'time_stop': '80% of time decay'
                    },
                    'position_sizing': {
                        'size': '1.5% of portfolio',
                        'max_risk': 'Strike price - premium received'
                    },
                    'rationale': [
                        f"Elevated volatility ({hist_vol*100:.1f}%) for premium selling",
                        f"Strike placed {(dist_to_low*50):.1f}% below market",
                        'Bullish bias with defined risk'
                    ]
                }
                suggestions.append(suggestion)
                
        else:  # Bearish trend
            if rsi > 30:  # Not oversold
                if dist_to_low > 0.03:  # Room to fall
                    suggestion = {
                        'strategy': 'Momentum Put Strategy',
                        'setup': {
                            'primary': f'Buy {symbol} Put Option',
                            'strike': round(current_price / 50) * 50,
                            'expiry': '1-2 months out',
                            'entry_levels': {
                                'ideal': round(current_price, 2),
                                'min': round(current_price * 0.99, 2)
                            }
                        },
                        'exit_plan': {
                            'target': round(current_price * (1 - dist_to_low), 2),
                            'stop_loss': round(current_price * (1 + dist_to_high * 0.5), 2),
                            'time_stop': '50% of time to expiry'
                        },
                        'position_sizing': {
                            'size': '2% of portfolio',
                            'adjustment': 'Reduce by 50%' if is_high_vol else 'Standard size'
                        },
                        'rationale': [
                            'Bearish trend confirmation',
                            f"RSI at {rsi:.1f} shows downward potential",
                            f"{(dist_to_low*100):.1f}% room to range low/support"
                        ]
                    }
                    suggestions.append(suggestion)
        
        # Add event-driven strategies
        if is_high_vol:
            if symbol == 'CRUDEOIL' and datetime.now().weekday() == 1:  # Tuesday
                suggestions.append({
                    'strategy': 'EIA Report Strangle',
                    'setup': {
                        'primary': [
                            f'Buy {symbol} {round(current_price/50)*50} Straddle',
                            'Exit before EIA report release'
                        ],
                        'expiry': '1-2 weeks out',
                        'entry_timing': 'Tuesday close for Wednesday report'
                    },
                    'rationale': [
                        'High volatility during inventory reports',
                        'Potential for large price swings',
                        'Short-term volatility play'
                    ],
                    'position_sizing': {
                        'size': '1% of portfolio',
                        'max_risk': 'Limited to premium paid'
                    }
                })
            elif symbol == 'NATURALGAS' and datetime.now().month in [11, 12, 1, 2]:
                suggestions.append({
                    'strategy': 'Weather Premium Calendar',
                    'setup': {
                        'primary': [
                            f'Buy {symbol} Call Calendar Spread',
                            f'Short front month {round(current_price/50)*50} Call',
                            f'Long next month {round(current_price/50)*50} Call'
                        ],
                        'rationale': 'Capture winter weather premium expansion'
                    },
                    'position_sizing': {
                        'size': '1.5% of portfolio',
                        'max_risk': 'Limited to spread cost'
                    }
                })
        
        logger.info(f"Generated {len(suggestions)} trading suggestions")
        logger.debug("Strategy types: " + ", ".join(s['strategy'] for s in suggestions))
        return suggestions
        
    except Exception as e:
        logger.error(f"Error generating trading suggestions: {str(e)}")
        return []

def validate_option_inputs(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> Tuple[bool, str]:
    """
    Validate option pricing inputs
    
    Args:
        S: Current price of underlying
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility of underlying
    
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        logger.debug(f"Validating inputs: S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
        
        if S <= 0:
            logger.warning(f"Invalid underlying price: {S}")
            return False, "Underlying price must be positive"
        
        if K <= 0:
            logger.warning(f"Invalid strike price: {K}")
            return False, "Strike price must be positive"
        
        if T < 0:
            logger.warning(f"Invalid time to expiry: {T}")
            return False, "Time to expiration cannot be negative"
        
        if r < -1 or r > 1:
            logger.warning(f"Invalid risk-free rate: {r}")
            return False, "Risk-free rate should be between -100% and 100%"
        
        if sigma < 0:
            logger.warning(f"Invalid volatility: {sigma}")
            return False, "Volatility cannot be negative"
        
        if sigma > 10:
            logger.warning(f"Unreasonably high volatility: {sigma}")
            return False, "Volatility seems unreasonably high (>1000%)"
        
        logger.debug("All inputs validated successfully")
        return True, "All inputs are valid"
        
    except Exception as e:
        return False, f"Error validating inputs: {str(e)}"
