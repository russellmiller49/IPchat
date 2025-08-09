#!/bin/bash

# Bronchmonkey Quick Start Script
echo "🐵 Starting Bronchmonkey..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env with your API keys and run this script again."
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check for required API key
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo "❌ Error: OPENAI_API_KEY not set in .env file"
    echo "Please add your OpenAI API key to the .env file"
    exit 1
fi

# Start PostgreSQL if on Mac with Homebrew
if command -v brew &> /dev/null; then
    echo "🗄️  Starting PostgreSQL..."
    brew services start postgresql@16
    sleep 2
fi

# Create database if it doesn't exist
if command -v psql &> /dev/null || command -v /opt/homebrew/opt/postgresql@16/bin/psql &> /dev/null; then
    PSQL_CMD="psql"
    if [ ! -x "$(command -v psql)" ]; then
        PSQL_CMD="/opt/homebrew/opt/postgresql@16/bin/psql"
    fi
    
    echo "📊 Setting up database..."
    $PSQL_CMD -lqt | cut -d \| -f 1 | grep -qw ip_rag || $PSQL_CMD -c "CREATE DATABASE ip_rag;"
    $PSQL_CMD -d ip_rag -f sql/schema.sql 2>/dev/null || echo "Schema already exists"
fi

# Start API server in background
echo "🚀 Starting API server..."
uvicorn backend.api.main:app --reload --port 8000 &
API_PID=$!

# Wait for API to be ready
echo "⏳ Waiting for API to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/docs > /dev/null; then
        echo "✅ API is ready!"
        break
    fi
    sleep 1
done

# Start Streamlit
echo "🎯 Starting Bronchmonkey UI..."
streamlit run chatbot_app.py --server.port 8501

# Cleanup on exit
trap "kill $API_PID 2>/dev/null" EXIT