#!/usr/bin/env python3
"""
Bronchmonkey Startup Script
Ensures everything is ready and starts the application
"""

import os
import sys
from pathlib import Path
import subprocess
import json

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print("✅ Python version OK")
    
    # Check .env file
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found")
        print("   Please copy .env.example to .env and add your OpenAI API key")
        return False
    
    # Check OpenAI API key
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_actual_api_key_here":
        print("❌ OPENAI_API_KEY not set in .env file")
        return False
    print("✅ OpenAI API key configured")
    
    # Check knowledge base
    indices_dir = Path("data/indices")
    if not indices_dir.exists():
        print("❌ Knowledge base not prepared")
        print("   Running preparation script...")
        try:
            subprocess.run([sys.executable, "prepare_knowledge_base.py"], check=True)
            print("✅ Knowledge base prepared")
        except:
            print("❌ Failed to prepare knowledge base")
            return False
    else:
        # Check if indices exist
        combined_kb = indices_dir / "combined_knowledge_base.json"
        if combined_kb.exists():
            with open(combined_kb, 'r') as f:
                kb = json.load(f)
            print(f"✅ Knowledge base ready ({kb['total_documents']} documents)")
        else:
            print("⚠️  Knowledge base incomplete, rebuilding...")
            subprocess.run([sys.executable, "prepare_knowledge_base.py"], check=True)
    
    # Check required packages
    required_packages = ['streamlit', 'openai', 'python-dotenv']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("   Installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages, check=True)
            print("✅ Packages installed")
        except:
            print("❌ Failed to install packages")
            return False
    else:
        print("✅ All packages installed")
    
    return True

def choose_app_version():
    """Let user choose which version to run"""
    print("\n📱 Choose Bronchmonkey version:")
    print("1. Lite Edition (simplified, faster)")
    print("2. Full Edition (advanced features)")
    
    # Default to Lite for testing
    choice = input("Enter choice [1]: ").strip() or "1"
    
    if choice == "1":
        return "bronchmonkey_lite.py"
    else:
        return "chatbot_app.py"

def start_app(app_file):
    """Start the Streamlit application"""
    print(f"\n🚀 Starting Bronchmonkey...")
    print(f"   Using: {app_file}")
    print("\n" + "="*50)
    print("🌐 Opening in browser: http://localhost:8501")
    print("📖 Press Ctrl+C to stop the application")
    print("="*50 + "\n")
    
    try:
        subprocess.run(["streamlit", "run", app_file])
    except KeyboardInterrupt:
        print("\n\n👋 Bronchmonkey stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return False
    
    return True

def main():
    print("="*50)
    print("🐵 BRONCHMONKEY STARTUP")
    print("="*50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please fix the issues above.")
        sys.exit(1)
    
    # Choose version
    app_file = choose_app_version()
    
    # Check if chosen file exists
    if not Path(app_file).exists():
        print(f"❌ {app_file} not found")
        if app_file == "bronchmonkey_lite.py":
            print("   Using chatbot_app.py instead...")
            app_file = "chatbot_app.py"
    
    # Start the app
    if not start_app(app_file):
        sys.exit(1)

if __name__ == "__main__":
    main()