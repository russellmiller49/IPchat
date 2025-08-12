# 🐵 Bronchmonkey - Interventional Pulmonology Research Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)

A sophisticated AI-powered research assistant designed specifically for **interventional pulmonology and critical care research**. Bronchmonkey combines hybrid search technology (vector, keyword, and SQL) with advanced language models to provide instant access to medical evidence from clinical trials, systematic reviews, and medical literature.

## 🚀 Quick Start

### Install and Run (Choose Your Edition)

```bash
# Clone the repository
git clone https://github.com/russellmiller49/IP_chat2.git
cd IP_chat2

# Install based on your needs
pip install -e ".[lite]"   # Lightweight, no database
pip install -e ".[full]"   # Full features with PostgreSQL
pip install -e ".[space]"  # Hugging Face Space deployment

# Set your OpenAI API key
export OPENAI_API_KEY=sk-...

# Run with the unified CLI
ipchat run --edition lite   # Start Bronchmonkey
```

Visit http://localhost:8501 to start querying medical evidence!

## 📦 Three Editions, One Codebase

### 🚀 **Lite Edition** (Recommended for most users)
- **Perfect for**: Personal use, demos, research
- **Features**: Local indexes, fast startup, no database required
- **Install**: `pip install -e ".[lite]"`
- **Run**: `ipchat run --edition lite`

### 🏢 **Full Edition** (Enterprise/Team use)
- **Perfect for**: Production deployments, research teams
- **Features**: PostgreSQL, API server, authentication, all features
- **Install**: `pip install -e ".[full]"`
- **Run**: `ipchat run --edition full`

### 🤗 **Space Edition** (Cloud deployment)
- **Perfect for**: Hugging Face Spaces, public demos
- **Features**: Pre-built indexes, basic auth, cloud-optimized
- **Deploy**: See [Space Deployment](#hugging-face-space-deployment)

## 🎯 Key Features

### Medical Evidence Search
- **Hybrid Search**: Combines FAISS vector search, BM25 keyword matching, and SQL queries
- **874 Document Chunks**: Granular evidence retrieval
- **292 Studies**: Comprehensive medical database
- **Smart Citations**: Automatic (Author Year) formatting with MLA bibliography

### AI-Powered Analysis
- **GPT-5 Ready**: Support for latest models including GPT-5
- **Depth Mode**: Comprehensive analysis with nuanced synthesis
- **Reduced Hallucinations**: Robust error handling and fact-checking
- **Context-Aware**: Specialized for medical terminology and research

### Professional Interface
- **Streamlit UI**: Clean, intuitive chat interface
- **Real-time Search**: Instant evidence retrieval
- **Citation Management**: Automatic bibliography generation
- **Export Ready**: Structured data output for further analysis

## 📊 Example Queries

Try these queries to explore the evidence base:

- "What percent of patients with BLVR had a pneumothorax?"
- "Compare robotic bronchoscopy diagnostic yields"
- "Show outcomes for central airway obstruction management"
- "FEV1 improvement with endobronchial valves at 12 months"
- "Adverse events in bronchial thermoplasty studies"

## 🛠️ Installation Options

### Option 1: Local Installation (Recommended)

```bash
# Clone and enter directory
git clone https://github.com/russellmiller49/IP_chat2.git
cd IP_chat2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package with desired edition
pip install -e ".[lite]"   # Most users
pip install -e ".[full]"   # Enterprise features
pip install -e ".[dev]"    # Development

# Set environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run
ipchat run --edition lite
```

### Option 2: Docker

```bash
# Lite Edition (single container)
docker-compose -f compose/docker-compose.lite.yml up

# Full Edition (with PostgreSQL)
docker-compose -f compose/docker-compose.full.yml up

# Build specific edition
docker build -f docker/Dockerfile.lite -t bronchmonkey:lite .
docker run -p 8501:8501 -e OPENAI_API_KEY=$OPENAI_API_KEY bronchmonkey:lite
```

### Option 3: One-Command Setup

```bash
# Quick setup script
curl -sSL https://raw.githubusercontent.com/russellmiller49/IP_chat2/main/setup.sh | bash
```

## 🎮 CLI Commands

The unified CLI provides comprehensive control:

```bash
# Run different editions
ipchat run --edition lite    # Lightweight version
ipchat run --edition full    # Full stack with API
ipchat run --edition space   # Hugging Face mode

# Utilities
ipchat info                  # Show configuration
ipchat verify                # Check installation
ipchat index --source data/  # Rebuild indexes

# Development
ipchat run --debug           # Debug mode
ipchat export --format json  # Export data
```

## 📁 Project Structure

```
IP_chat2/
├── ipchat/                  # Main package
│   ├── core/               # Shared logic (all editions)
│   │   ├── retrieval/      # Search algorithms
│   │   ├── pipelines/      # RAG orchestration
│   │   ├── citations/      # Citation formatting
│   │   └── config/         # Configuration management
│   ├── adapters/           # External service adapters
│   │   └── llm/           # LLM providers (OpenAI, etc.)
│   ├── apps/              # Edition-specific applications
│   └── cli.py             # Command-line interface
├── data/                   # Indexes and evidence
├── docker/                 # Docker configurations
├── compose/               # Docker Compose files
└── docs/                  # Documentation
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
IPCHAT_EDITION=lite              # Default edition
IPCHAT_LLM__MODEL=gpt-5-mini    # AI model
IPCHAT_DEPTH_FEATURES=true      # Enable depth mode
IPCHAT_DEBUG_MODE=false         # Debug logging
```

### Configuration File

Create `config.json` for persistent settings:

```json
{
  "edition": "lite",
  "llm": {
    "model": "gpt-5-mini",
    "temperature": 0.3
  },
  "retrieval": {
    "num_results": 10,
    "vector_weight": 0.5,
    "bm25_weight": 0.3
  }
}
```

## 🚢 Deployment

### Hugging Face Space Deployment

```bash
# Build Space Docker image
docker build -f docker/Dockerfile.space -t bronchmonkey:space .

# Test locally on Space port
docker run -p 7860:7860 -e BASIC_AUTH_USERS=user:pass bronchmonkey:space

# Deploy to Hugging Face
huggingface-cli upload bronchmonkey ./space --repo-type space
```

### Production Deployment (Full Edition)

```bash
# Start full stack with PostgreSQL
docker-compose -f compose/docker-compose.full.yml up -d

# Check health
curl http://localhost:8000/health  # API
curl http://localhost:8501/_stcore/health  # UI
```

## 📊 Feature Comparison

| Feature | Lite | Full | Space |
|---------|:----:|:----:|:-----:|
| **Streamlit UI** | ✅ | ✅ | ✅ |
| **FastAPI Server** | ❌ | ✅ | ❌ |
| **PostgreSQL** | ❌ | ✅ | ❌ |
| **FAISS Search** | ✅ | ✅ | ✅ |
| **BM25 Search** | ✅ | ✅ | ✅ |
| **Depth Mode** | ✅ | ✅ | ✅ |
| **Authentication** | Optional | ✅ | ✅ |
| **Docker Support** | ✅ | ✅ | ✅ |
| **Cloud Ready** | ✅ | ✅ | ✅ |

## 🧪 Development

### Setup Development Environment

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Code formatting
black ipchat/
ruff check ipchat/

# Type checking
mypy ipchat/
```

### Adding New Features

1. Implement in `ipchat/core/` for shared logic
2. Add edition-specific behavior in config
3. Update tests in `tests/`
4. Document in `docs/`

## 📚 Documentation

- [Migration Guide](MIGRATION.md) - Upgrading from old versions
- [Unified Architecture](docs/UNIFIED_ARCHITECTURE.md) - Technical details
- [API Reference](docs/API.md) - API endpoints and usage
- [Contributing](CONTRIBUTING.md) - How to contribute

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [OpenAI GPT](https://openai.com/)
- Search by [FAISS](https://github.com/facebookresearch/faiss)
- Medical evidence from peer-reviewed publications

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/russellmiller49/IP_chat2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/russellmiller49/IP_chat2/discussions)
- **Email**: support@bronchmonkey.ai

---

**Bronchmonkey** - Evidence-based medicine at your fingertips 🐵

*Version 0.2.0 | Last Updated: November 2024*