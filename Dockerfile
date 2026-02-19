FROM python:3.12-slim

# Install system dependencies required by CairoSVG and OpenCV
RUN apt-get update && apt-get install -y \
    libcairo2 \
    libpango-1.0-0 \
    libgdk-pixbuf-xlib-2.0-0 \
    libffi-dev \
    libgl1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements_webapp.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_webapp.txt

# Copy project files
COPY . .

# Flask port
EXPOSE 8000

# Start Flask app
CMD ["python", "app.py"]
