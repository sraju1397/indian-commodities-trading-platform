@echo off
REM Indian Commodities Options Trading Platform - Windows Setup Script
REM This script creates a virtual environment, installs dependencies, and runs the app

echo 🚀 Setting up Indian Commodities Options Trading Platform...
echo =================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo [INFO] Python found, proceeding with setup...

REM Create virtual environment if it doesn't exist
if not exist "trading_env" (
    echo [INFO] Creating virtual environment...
    python -m venv trading_env
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created successfully
) else (
    echo [INFO] Virtual environment already exists
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call trading_env\Scripts\activate.bat

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Create requirements file if it doesn't exist
if not exist "requirements.txt" (
    echo [INFO] Creating requirements.txt...
    (
        echo streamlit^>=1.28.0
        echo pandas^>=1.5.0
        echo plotly^>=5.15.0
        echo yfinance^>=0.2.18
        echo scipy^>=1.10.0
        echo sqlalchemy^>=2.0.0
        echo psycopg2-binary^>=2.9.0
        echo pytz^>=2023.3
        echo requests^>=2.31.0
        echo numpy^>=1.24.0
    ) > requirements.txt
    echo [SUCCESS] requirements.txt created
)

REM Install dependencies
echo [INFO] Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Failed to install some dependencies
    pause
    exit /b 1
)

echo [SUCCESS] All dependencies installed successfully

REM Create .streamlit directory and config if they don't exist
if not exist ".streamlit" (
    echo [INFO] Creating Streamlit configuration...
    mkdir .streamlit
    (
        echo [server]
        echo headless = true
        echo address = "0.0.0.0"
        echo port = 5000
        echo.
        echo [theme]
        echo primaryColor = "#FF6B35"
        echo backgroundColor = "#FFFFFF"
        echo secondaryBackgroundColor = "#F0F2F6"
        echo textColor = "#262730"
    ) > .streamlit\config.toml
    echo [SUCCESS] Streamlit configuration created
)

REM Database setup information
echo.
echo [WARNING] IMPORTANT: PostgreSQL Database Setup
echo ----------------------------------------
echo This application requires a PostgreSQL database to store your trading data.
echo.
echo Options for PostgreSQL setup:
echo 1. Local installation (PostgreSQL for Windows)
echo 2. Cloud database (Supabase, ElephantSQL, etc.)
echo 3. Docker Desktop with PostgreSQL container
echo.
echo For Windows, download PostgreSQL from: https://www.postgresql.org/download/windows/
echo.
echo See README.md for detailed PostgreSQL setup instructions.
echo.

REM Check if DATABASE_URL is set
if "%DATABASE_URL%"=="" (
    echo [WARNING] DATABASE_URL environment variable not set.
    echo The app will try to connect to a local PostgreSQL database.
    echo If you don't have PostgreSQL set up, some features may not work.
    echo.
)

REM Start the application
echo [INFO] Starting the application...
echo.
echo [SUCCESS] 🎉 Setup complete! Starting the Indian Commodities Options Trading Platform...
echo.
echo The application will be available at: http://localhost:5000
echo Press Ctrl+C to stop the application
echo.

REM Run the Streamlit app
streamlit run app.py --server.port 5000

REM Deactivate virtual environment when done
call deactivate
pause