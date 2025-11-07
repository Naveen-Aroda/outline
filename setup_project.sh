#!/usr/bin/env bash
# ============================================================
# Linux project setup script (Ubuntu/Debian based)
# Installs Python 3.11, pip, virtualenv, Inkscape, Potrace,
# and project requirements.
# ============================================================

set -e  # exit immediately if a command fails

# ---- Step 1: Ensure system packages ----
echo "🔧 Updating package list..."
sudo apt update -y

echo "📦 Installing Python 3.11 and tools..."
sudo apt install -y python3.11 python3.11-venv python3.11-distutils curl wget

# ---- Step 2: Ensure pip for Python 3.11 ----
echo "🐍 Ensuring pip for Python 3.11..."
python3.11 -m ensurepip --upgrade || true
python3.11 -m pip install --upgrade pip setuptools wheel

# ---- Step 3: Create virtual environment ----
echo "🌱 Creating virtual environment..."
python3.11 -m venv .venv

# ---- Step 4: Fix permissions (optional safety) ----
echo "🔒 Fixing venv ownership..."
sudo chown -R "$USER:$USER" .venv

# ---- Step 5: Activate venv ----
echo "⚙️ Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate



# ---- Step 6: Install Python requirements ----
if [ -f "requirements.txt" ]; then
    echo "📜 Installing Python requirements..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
else
    echo "⚠️  No requirements.txt found — skipping Python deps."
fi

# ---- Step 7: Confirm success ----
echo "✅ Setup complete!"
echo
echo "To activate your environment later, run:"
echo "  source .venv/bin/activate"
echo
echo "Then run your scripts with:"
echo "  python process_svg.py"
