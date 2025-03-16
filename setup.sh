#!/bin/bash

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create uploads directory if it doesn't exist
echo "Creating uploads directory..."
mkdir -p uploads

# Check if Docker is installed
if command -v docker &> /dev/null; then
    echo "Starting Qdrant with Docker Compose..."
    docker-compose up -d
else
    echo "Docker not found. Please install Docker and Docker Compose to run Qdrant."
    echo "You can download Docker from https://www.docker.com/get-started"
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "Please edit the .env file with your AWS credentials."
fi

# Check if poppler is installed
if command -v pdfinfo &> /dev/null; then
    echo "Poppler is installed. PDF processing will use enhanced features."
else
    echo "Poppler not found. PDF processing will use basic features."
    echo "To install Poppler:"
    echo "  - On macOS: brew install poppler"
    echo "  - On Ubuntu/Debian: sudo apt-get install poppler-utils"
    echo "  - On Windows with Chocolatey: choco install poppler"
fi

echo ""
echo "Setup complete! To start the application, run:"
echo "  source venv/bin/activate  # If not already activated"
echo "  python app.py"
echo ""
echo "Then visit http://localhost:8000/docs to see the API documentation." 