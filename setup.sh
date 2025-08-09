#!/bin/bash

# Bronchmonkey Initial Setup Script
echo "🐵 Welcome to Bronchmonkey Setup!"
echo "================================="

# Function to check command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
echo "🐍 Checking Python..."
if ! command_exists python3; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

# Check pip
echo "📦 Checking pip..."
if ! command_exists pip3 && ! command_exists pip; then
    echo "❌ pip is required but not installed."
    echo "Installing pip..."
    python3 -m ensurepip --default-pip
fi

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip3 install -r requirements.txt

# Setup environment file
if [ ! -f .env ]; then
    echo "🔐 Setting up environment configuration..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env file with your API keys:"
    echo "   - OPENAI_API_KEY (required)"
    echo "   - GEMINI_API_KEY (optional)"
    echo ""
    read -p "Press Enter to continue after adding your API keys..."
fi

# PostgreSQL setup
echo "🗄️  Setting up PostgreSQL..."
if command_exists psql || [ -f /opt/homebrew/opt/postgresql@16/bin/psql ]; then
    PSQL_CMD="psql"
    if [ ! -x "$(command -v psql)" ]; then
        PSQL_CMD="/opt/homebrew/opt/postgresql@16/bin/psql"
    fi
    
    # Create database
    echo "Creating database..."
    $PSQL_CMD -c "CREATE DATABASE ip_rag;" 2>/dev/null || echo "Database already exists"
    
    # Load schema
    echo "Loading schema..."
    $PSQL_CMD -d ip_rag -f sql/schema.sql 2>/dev/null || echo "Schema already loaded"
    
    # Load data
    echo "Loading research data..."
    export DATABASE_URL="postgresql://$(whoami)@localhost/ip_rag"
    python3 ingestion/load_json_to_pg.py --trials-dir data/oe_final_outputs
else
    echo "⚠️  PostgreSQL not found. Using file-based storage only."
    echo "For full functionality, install PostgreSQL:"
    echo "  Mac: brew install postgresql@16"
    echo "  Ubuntu: sudo apt-get install postgresql"
fi

# Build search indexes
echo "🔍 Building search indexes..."

# Check if chunks exist
if [ ! -f data/chunks/chunks.jsonl ]; then
    echo "Creating document chunks..."
    python3 chunking/chunker.py --trials-dir data/oe_final_outputs
fi

# Build BM25 index
if [ ! -f data/index/bm25.pkl ]; then
    echo "Building keyword search index..."
    python3 indexing/build_bm25.py
fi

# Build FAISS index
if [ ! -f data/index/faiss.index ]; then
    echo "Building vector search index..."
    python3 indexing/build_faiss.py
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start Bronchmonkey, run:"
echo "   ./start.sh"
echo ""
echo "Or manually start with:"
echo "   # Terminal 1: Start API"
echo "   uvicorn backend.api.main:app --reload --port 8000"
echo ""
echo "   # Terminal 2: Start UI"
echo "   streamlit run chatbot_app.py"
echo ""
echo "Then open http://localhost:8501 in your browser"
echo ""
echo "🐵 Happy researching with Bronchmonkey!"