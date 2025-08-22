#!/usr/bin/env python3
"""Simple GPT-5 test"""

import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not found in .env file")
    exit(1)

try:
    import openai
    print("Testing GPT-5 models...")
    
    client = openai.OpenAI()
    
    # Test GPT-5
    print("\n1. Testing gpt-5:")
    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        print(f"   ✅ SUCCESS with max_tokens")
        print(f"   Model: {response.model}")
    except Exception as e:
        if "max_completion_tokens" in str(e):
            try:
                response = client.chat.completions.create(
                    model="gpt-5",
                    messages=[{"role": "user", "content": "Say hello"}],
                    max_completion_tokens=10
                )
                print(f"   ✅ SUCCESS with max_completion_tokens")
                print(f"   Model: {response.model}")
            except Exception as e2:
                print(f"   ❌ Failed: {str(e2)[:100]}")
        else:
            print(f"   ❌ Failed: {str(e)[:100]}")
    
    # Test GPT-5-mini
    print("\n2. Testing gpt-5-mini:")
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        print(f"   ✅ SUCCESS with max_tokens")
        print(f"   Model: {response.model}")
    except Exception as e:
        if "max_completion_tokens" in str(e):
            try:
                response = client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[{"role": "user", "content": "Say hello"}],
                    max_completion_tokens=10
                )
                print(f"   ✅ SUCCESS with max_completion_tokens")
                print(f"   Model: {response.model}")
            except Exception as e2:
                print(f"   ❌ Failed: {str(e2)[:100]}")
        else:
            print(f"   ❌ Failed: {str(e)[:100]}")
            
    print("\nNote: If models show as 'not found', your API key may not have GPT-5 access yet.")
    
except ImportError:
    print("OpenAI library not installed")