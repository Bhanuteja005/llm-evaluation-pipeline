#!/bin/bash
# Setup script for LLM Evaluation Pipeline

set -e

echo "🚀 Setting up LLM Evaluation Pipeline..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "✓ Virtual environment activated"

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Copy environment file
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "✓ .env file created (please edit with your settings)"
else
    echo "✓ .env file already exists"
fi

# Create output directory
mkdir -p output
echo "✓ Output directory created"

# Download sample files if needed
mkdir -p samples
echo "✓ Samples directory ready"

# Run tests
echo "🧪 Running tests..."
pytest tests/ -v

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env file with your configuration"
echo "  2. Run: python -m src.cli evaluate -c <conversation.json> -x <context.json>"
echo "  3. Or run with samples: python -m src.cli evaluate -c Json/sample-chat-conversation-01.json -x Json/sample_context_vectors-01.json"
echo ""
