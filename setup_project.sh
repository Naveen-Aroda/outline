#!/bin/bash
set -e

echo "🚀 Installing Homebrew (if not installed)..."
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

echo "🍺 Installing dependencies (python, cairo, libffi)..."
brew install python libffi cairo

echo "🐍 Creating virtual environment using Homebrew Python..."
/opt/homebrew/bin/python3 -m venv .venv

echo "📦 Installing Python dependencies..."
.venv/bin/pip install --upgrade pip
if [ -f requirements.txt ]; then
  .venv/bin/pip install -r requirements.txt
else
  .venv/bin/pip install cairosvg
fi

echo "🧪 Testing CairoSVG installation..."
.venv/bin/cairosvg --help

echo "✅ Setup complete! Use 'source .venv/bin/activate' to start."