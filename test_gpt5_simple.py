#!/usr/bin/env python3
"""
Simple GPT-5 API connection test
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not found in environment!")
    sys.exit(1)

print("Testing GPT-5 API connection...")
print("-" * 50)

client = OpenAI()

# Test 1: Simple text response
print("\n1. Testing basic GPT-5 response...")
try:
    response = client.responses.create(
        model="gpt-5",
        input="Return the JSON object: {\"test\": \"success\", \"model\": \"gpt-5\"}",
        max_output_tokens=50
    )
    print(f"   ✅ Success! Response: {response.output_text}")
except Exception as e:
    print(f"   ❌ Failed: {type(e).__name__}: {str(e)[:200]}")

# Test 2: JSON extraction (like the textbook extractor)
print("\n2. Testing JSON extraction...")
try:
    prompt = """Extract the following as JSON:
    Title: Test Chapter
    Authors: John Doe, Jane Smith
    
    Return as JSON with keys: title, authors (as array)"""
    
    response = client.responses.create(
        model="gpt-5",
        input=prompt,
        max_output_tokens=100
    )
    print(f"   ✅ Success! Response: {response.output_text}")
except Exception as e:
    print(f"   ❌ Failed: {type(e).__name__}: {str(e)[:200]}")

# Test 3: Check model availability
print("\n3. Checking available models...")
try:
    models = client.models.list()
    gpt5_models = [m.id for m in models if 'gpt-5' in m.id.lower()]
    if gpt5_models:
        print(f"   ✅ GPT-5 models available: {', '.join(gpt5_models)}")
    else:
        print("   ⚠️ No GPT-5 models found in account")
        all_models = [m.id for m in models]
        print(f"   Available models: {', '.join(all_models[:10])}...")
except Exception as e:
    print(f"   ❌ Failed to list models: {e}")

print("\n" + "-" * 50)
print("Test complete!")