FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for genomic analysis
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libhdf5-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/external /app/data/interim \
    /app/models/trained /app/models/artifacts /app/logs /app/mlruns

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Expose ports
EXPOSE 8000 5000

# Default command
CMD ["python", "scripts/train.py", "--help"]