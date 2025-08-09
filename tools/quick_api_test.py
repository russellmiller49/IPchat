#!/usr/bin/env python3
"""Quick test to verify OpenAI API connection."""

import os
import sys
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)
    print(f"✓ Loaded .env from {env_path}")

# Check API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("✗ No OPENAI_API_KEY found in environment")
    sys.exit(1)

print(f"✓ Found API key: {api_key[:10]}...")

# Test connection
try:
    from openai import OpenAI
    client = OpenAI()
    
    print("\nTesting API connection...")
    
    # Use a simple model first to test
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say 'API working'"}],
        max_tokens=10,
        timeout=10  # 10 second timeout
    )
    
    result = response.choices[0].message.content
    print(f"✓ API Response: {result}")
    
    # Now test GPT-5
    print("\nTesting GPT-5 model...")
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": "Say 'GPT-5 working'"}],
        max_completion_tokens=10,  # GPT-5 uses max_completion_tokens
        timeout=10
    )
    
    result = response.choices[0].message.content
    print(f"✓ GPT-5 Response: {result}")
    
except Exception as e:
    print(f"✗ API Error: {e}")
    print("\nPossible issues:")
    print("1. Invalid API key")
    print("2. Network/firewall blocking OpenAI")
    print("3. GPT-5 not available on your account yet")
    print("4. Rate limits exceeded")
    sys.exit(1)

print("\n✓ All tests passed! API is working correctly.")