# Setup script for Windows
# Run with: powershell -ExecutionPolicy Bypass -File setup.ps1

Write-Host "🚀 Setting up LLM Evaluation Pipeline..." -ForegroundColor Cyan

# Check Python version
$pythonVersion = python --version 2>&1
Write-Host "✓ Python version: $pythonVersion" -ForegroundColor Green

# Create virtual environment
Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "✓ Activating virtual environment..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "📦 Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Copy environment file
if (-not (Test-Path .env)) {
    Write-Host "📝 Creating .env file..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "✓ .env file created (please edit with your settings)" -ForegroundColor Green
} else {
    Write-Host "✓ .env file already exists" -ForegroundColor Green
}

# Create output directory
New-Item -ItemType Directory -Force -Path output | Out-Null
Write-Host "✓ Output directory created" -ForegroundColor Green

# Create samples directory
New-Item -ItemType Directory -Force -Path samples | Out-Null
Write-Host "✓ Samples directory ready" -ForegroundColor Green

# Run tests
Write-Host "🧪 Running tests..." -ForegroundColor Yellow
pytest tests\ -v

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Edit .env file with your configuration"
Write-Host "  2. Run: python -m src.cli evaluate -c <conversation.json> -x <context.json>"
Write-Host "  3. Or run with provided samples"
Write-Host ""
