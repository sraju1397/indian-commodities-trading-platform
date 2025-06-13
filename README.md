# Indian Commodities Options Trading Platform 📊

A comprehensive Streamlit-based trading platform for analyzing Indian commodities options with real-time data, intelligent recommendations, and portfolio management.

## 🚀 Features

- **Real-time Market Data**: Live commodity prices and charts for Gold, Silver, Crude Oil, Natural Gas, and Copper
- **Options Chain Analysis**: Complete options pricing with Greeks calculations
- **Intelligent Recommendations**: AI-powered trading suggestions with risk levels and confidence scores
- **Advanced Screener**: Filter options based on technical indicators and criteria
- **Portfolio Management**: Track positions, P&L, and performance metrics
- **Risk Dashboard**: Comprehensive risk analysis with stress testing
- **Database Integration**: PostgreSQL backend for persistent data storage

## 📋 Prerequisites

- Python 3.8 or higher
- PostgreSQL database (local or cloud)
- Internet connection for market data

## 🛠️ Installation

### Quick Start (Automated Setup)

For the fastest setup, use our automated installation script:

```bash
# Clone the repository
git clone <your-repository-url>
cd indian-commodities-trading-platform

# Run the setup script (creates virtual environment, installs dependencies, runs app)
chmod +x setup_and_run.sh
./setup_and_run.sh
```

### Manual Installation

#### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd indian-commodities-trading-platform
```

#### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv trading_env

# Activate virtual environment
# On macOS/Linux:
source trading_env/bin/activate
# On Windows:
# trading_env\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install streamlit pandas plotly yfinance scipy sqlalchemy psycopg2-binary pytz requests
```

### 3. Set Up PostgreSQL Database

#### Option A: Local Installation (MacBook Pro M1)

**Using Homebrew (Recommended):**

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install PostgreSQL
brew install postgresql@15

# Start PostgreSQL service
brew services start postgresql@15

# Create database
createdb trading_platform_db
```

**Using Postgres.app:**

1. Download from https://postgresapp.com/
2. Install and run the application
3. Create a new database named `trading_platform_db`

**Using Docker:**

```bash
docker run --name postgres-trading \
  -e POSTGRES_PASSWORD=yourpassword \
  -e POSTGRES_DB=trading_platform_db \
  -p 5432:5432 \
  -d postgres:15
```

#### Option B: Cloud Database

- Use services like Supabase, ElephantSQL, or AWS RDS
- Get your connection URL in the format: `postgresql://username:password@host:port/database`

### 4. Environment Configuration

Create a `.env` file in the project root (optional):

```env
DATABASE_URL=postgresql://username:password@localhost:5432/trading_platform_db
PGHOST=localhost
PGPORT=5432
PGUSER=postgres
PGPASSWORD=yourpassword
PGDATABASE=trading_platform_db
```

## 🏃‍♂️ Running the Application

### Using the Setup Script (Recommended)

```bash
./setup_and_run.sh
```

### Manual Start

```bash
# Activate virtual environment first (if using manual installation)
source trading_env/bin/activate

# Start the Streamlit server
streamlit run app.py --server.port 5000
```

The application will be available at: http://localhost:5000

### For Subsequent Runs

```bash
# Just activate environment and run (no need to reinstall)
source trading_env/bin/activate
streamlit run app.py --server.port 5000
```

### First Time Setup

1. The database tables will be created automatically on first run
2. Navigate through the different pages to explore features
3. Start by checking Market Data for real-time commodity prices

## 📖 User Guide

### 1. 📊 Market Data

- View real-time commodity prices
- Interactive candlestick charts
- Technical indicators (RSI, MACD, Bollinger Bands)
- Market status and session information

### 2. ⚡ Options Chain

- Complete options pricing for selected commodities
- Greeks calculations (Delta, Gamma, Theta, Vega)
- Implied volatility analysis
- Strike price analysis

### 3. 🎯 Recommendations

- AI-powered trading suggestions
- Multiple strategy recommendations:
  - Bull Call Spreads
  - Bear Put Spreads
  - Iron Condors
  - Covered Calls
  - Protective Puts
- Risk levels and confidence scores
- Detailed reasoning for each recommendation

### 4. 🔍 Screener

- Filter options based on:
  - Moneyness (ITM, ATM, OTM)
  - Volume and Open Interest
  - Technical indicators
  - Greeks values
- Custom screening criteria

### 5. 💼 Portfolio

- Add and manage options positions
- Real-time P&L tracking
- Performance analytics
- Trade history and notes

### 6. ⚠️ Risk Dashboard

- Portfolio risk metrics
- Stress testing scenarios
- Position sizing recommendations
- Risk-adjusted returns

## 🗄️ Database Schema

The application uses PostgreSQL with the following main tables:

### portfolio_positions

Stores all options trading positions

```sql
- id (Primary Key)
- user_id (Default: 'default_user')
- commodity (Gold, Silver, etc.)
- option_type (Call/Put)
- action (Buy/Sell)
- strike_price, quantity, premium
- expiry_date, trade_date
- status (Active/Closed)
- notes, entry_time
```

### recommendations_history

Tracks all generated recommendations

```sql
- id (Primary Key)
- commodity, strategy_name
- action, strike_price, expiry
- premium, max_loss, target_profit
- risk_level, confidence_score
- market_bias, reasoning
- created_at, user_id
```

### market_data_cache

Caches commodity price data

```sql
- id (Primary Key)
- symbol, date
- open_price, high_price, low_price, close_price
- volume, updated_at
```

## 🔧 Configuration

### Streamlit Configuration

The `.streamlit/config.toml` file contains:

```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

### Default Commodities

The platform focuses on the most liquid Indian commodity markets:

- **Gold** (GOLD=F)
- **Silver** (SI=F)
- **Crude Oil** (CL=F)
- **Natural Gas** (NG=F)
- **Copper** (HG=F)

## 📊 Data Sources

- **Market Data**: Yahoo Finance API via yfinance library
- **Options Pricing**: Black-Scholes model with real-time calculations
- **Technical Analysis**: Custom indicators and scipy-based calculations

## 🚨 Important Notes

### PostgreSQL Benefits

- **Free and Open Source**: No licensing costs
- **Reliable**: ACID compliant with excellent data integrity
- **Scalable**: Handles large datasets efficiently
- **Cross-platform**: Works on Windows, macOS, and Linux

### Data Accuracy

- Market data is fetched from Yahoo Finance
- Options pricing uses industry-standard Black-Scholes model
- Some commodities may have limited historical data
- Always verify trades with your broker before execution

### Risk Disclaimer

This platform is for educational and analysis purposes only. Always:

- Consult with financial advisors
- Verify all data independently
- Understand options trading risks
- Never risk more than you can afford to lose

## 🔧 Troubleshooting

### Database Connection Issues

```bash
# Check if PostgreSQL is running
brew services list | grep postgresql

# Restart PostgreSQL service
brew services restart postgresql@15

# Test connection
psql -d trading_platform_db
```

### Common Errors

1. **ModuleNotFoundError**: Install missing dependencies with pip
2. **Database connection failed**: Check DATABASE_URL and PostgreSQL service
3. **Data not loading**: Verify internet connection for market data

## 📁 Git Repository Setup

### Initial Setup

When setting up your local repository:

```bash
# Initialize Git repository (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Indian Commodities Options Trading Platform"

# Add remote repository (replace with your GitHub/GitLab URL)
git remote add origin https://github.com/yourusername/indian-commodities-trading.git

# Push to remote
git push -u origin main
```

### Important Files

- `.gitignore` - Protects sensitive data and environment files
- `setup_and_run.sh` - Automated setup script for macOS/Linux
- `setup_and_run.bat` - Automated setup script for Windows
- `README.md` - Comprehensive documentation

### Repository Structure

```
indian-commodities-trading/
├── pages/                  # Streamlit pages
├── utils/                  # Core utilities and logic
├── .streamlit/            # Streamlit configuration
├── setup_and_run.sh       # Setup script (Unix)
├── setup_and_run.bat      # Setup script (Windows)
├── app.py                 # Main application
├── README.md              # Documentation
└── .gitignore            # Git ignore rules
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review error logs in the terminal
3. Ensure all dependencies are installed
4. Verify database connectivity

---

**Happy Trading! 🚀📈**

_Remember: This platform is for educational purposes. Always do your own research and consult financial advisors before making trading decisions._

-- myuser
-- your_password

# If you have psql installed locally

psql -h localhost -p 5432 -U postgres -d trading_db
When prompted, use the password: your_password

Connecting Your Trading Platform Locally
Once PostgreSQL is running locally, you'll need to update your connection settings:

Database URL format: postgresql://username:password@localhost:5432/database_name
Default credentials: Usually username: postgres, password: (whatever you set)
Port: Default is 5432
