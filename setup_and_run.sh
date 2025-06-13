#!/bin/bash

# Indian Commodities Options Trading Platform - Setup and Run Script
# This script creates a virtual environment, installs dependencies, and runs the app

echo "🚀 Setting up Indian Commodities Options Trading Platform..."
echo "================================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
print_status "Found Python $PYTHON_VERSION"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install pip3."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "trading_env" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv trading_env
    if [ $? -eq 0 ]; then
        print_success "Virtual environment created successfully"
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source trading_env/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    print_status "Creating requirements.txt..."
    cat > requirements.txt << EOF
streamlit>=1.28.0
pandas>=1.5.0
plotly>=5.15.0
yfinance>=0.2.18
scipy>=1.10.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
pytz>=2023.3
requests>=2.31.0
numpy>=1.24.0
EOF
    print_success "requirements.txt created"
fi

# Install dependencies
print_status "Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    print_success "All dependencies installed successfully"
else
    print_error "Failed to install some dependencies"
    exit 1
fi

# Create .streamlit directory and config if they don't exist
if [ ! -d ".streamlit" ]; then
    print_status "Creating Streamlit configuration..."
    mkdir -p .streamlit
    cat > .streamlit/config.toml << EOF
[server]
headless = true
address = "0.0.0.0"
port = 3000

[theme]
primaryColor = "#FF6B35"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
EOF
    print_success "Streamlit configuration created"
fi

# Check if PostgreSQL is needed
echo ""
print_warning "IMPORTANT: PostgreSQL Database Setup"
echo "----------------------------------------"
echo "This application requires a PostgreSQL database to store your trading data."
echo ""
echo "Options for PostgreSQL setup:"
echo "1. Local installation (recommended for development)"
echo "2. Cloud database (Supabase, ElephantSQL, etc.)"
echo "3. Docker container"
echo ""
echo "For MacBook Pro M1, the easiest local setup is:"
echo "  brew install postgresql@15"
echo "  brew services start postgresql@15"
echo "  createdb trading_platform_db"
echo ""
echo "See README.md for detailed PostgreSQL setup instructions."
echo ""

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    print_warning "DATABASE_URL environment variable not set."
    echo "The app will try to connect to a local PostgreSQL database."
    echo "If you don't have PostgreSQL set up, some features may not work."
    echo ""
fi

# Start the application
print_status "Starting the application..."
echo ""
print_success "🎉 Setup complete! Starting the Indian Commodities Options Trading Platform..."
echo ""
echo "The application will be available at: http://localhost:3000"
echo "Press Ctrl+C to stop the application"
echo ""

# Run the Streamlit app
streamlit run app.py --server.port 3000

# Deactivate virtual environment when done
deactivate