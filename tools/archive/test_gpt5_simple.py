#!/usr/bin/env python3
"""Simple test of GPT-5 with minimal JSON output."""

import os
import sys
import json
from pathlib import Path

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)

# Simple schema
simple_schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "year": {"type": "integer"}
    },
    "required": ["title", "year"]
}

client = OpenAI()

print("Testing GPT-5 with simple JSON generation...")
print(f"API Key: {os.getenv('OPENAI_API_KEY')[:20]}...")

try:
    # Test 1: Simple JSON without schema
    print("\n1. Testing basic JSON response...")
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You are a JSON generator. Output only valid JSON."},
            {"role": "user", "content": 'Generate JSON with title="Test" and year=2024'}
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=100,
        timeout=30
    )
    
    result = response.choices[0].message.content
    print(f"✓ Response received: {result}")
    parsed = json.loads(result)
    print(f"✓ Valid JSON with keys: {list(parsed.keys())}")
    
    # Test 2: JSON with schema
    print("\n2. Testing JSON with schema...")
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "Generate JSON matching the schema."},
            {"role": "user", "content": "Create a document with title 'Medical Paper' and year 2024"}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "simple_test",
                "schema": simple_schema,
                "strict": False
            }
        },
        max_completion_tokens=100,
        timeout=30
    )
    
    result = response.choices[0].message.content
    print(f"✓ Response received: {result}")
    parsed = json.loads(result)
    print(f"✓ Valid JSON: title='{parsed.get('title')}', year={parsed.get('year')}")
    
    print("\n✓ All tests passed! GPT-5 JSON generation is working.")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check if GPT-5 is available on your account")
    print("2. Try with gpt-4-turbo-preview instead")
    print("3. Check API rate limits")
    print("4. Verify network connectivity")