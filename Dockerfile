# Multi-stage build for Bronchmonkey
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (if needed)
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" || true

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/index data/chunks data/oe_final_outputs

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Start both services using a shell script
CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 & streamlit run chatbot_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"]