#!/usr/bin/env python3
"""
Test direct GPT-5 model usage without fallbacks
"""

import os
from dotenv import load_dotenv
import openai

load_dotenv()

def test_gpt5_models():
    """Test GPT-5 models directly"""
    
    client = openai.OpenAI()
    
    # Test configurations
    test_configs = [
        {
            "name": "GPT-5-mini with max_tokens",
            "model": "gpt-5-mini",
            "params": {"max_tokens": 100}
        },
        {
            "name": "GPT-5 with max_tokens",
            "model": "gpt-5",
            "params": {"max_tokens": 100}
        },
        {
            "name": "GPT-5-mini with max_completion_tokens",
            "model": "gpt-5-mini",
            "params": {"max_completion_tokens": 100}
        },
        {
            "name": "GPT-5 with max_completion_tokens",
            "model": "gpt-5",
            "params": {"max_completion_tokens": 100}
        }
    ]
    
    print("=" * 60)
    print("Testing Direct GPT-5 Model Access")
    print("=" * 60)
    
    for config in test_configs:
        print(f"\n{config['name']}:")
        print("-" * 40)
        
        try:
            response = client.chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the diagnostic yield of EBUS-TBNA? Answer in one sentence."}
                ],
                temperature=0.2,
                **config["params"]
            )
            
            print(f"✅ SUCCESS!")
            print(f"   Model used: {response.model}")
            print(f"   Response: {response.choices[0].message.content[:200]}")
            
        except Exception as e:
            error_msg = str(e)
            if "model" in error_msg.lower() and "not found" in error_msg.lower():
                print(f"❌ Model not available: {config['model']}")
            elif "max_tokens" in error_msg and "max_completion_tokens" in error_msg:
                print(f"⚠️  Wrong parameter - should use max_completion_tokens")
            elif "max_completion_tokens" in error_msg and "max_tokens" in error_msg:
                print(f"⚠️  Wrong parameter - should use max_tokens")
            else:
                print(f"❌ Error: {error_msg[:200]}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("-" * 40)
    print("Based on the results above:")
    print("1. If GPT-5 models show as 'not available', your API key may not have access")
    print("2. Check which parameter format works (max_tokens vs max_completion_tokens)")
    print("3. Use the successful configuration in bronchmonkey_gpt5.py")
    print("=" * 60)

if __name__ == "__main__":
    test_gpt5_models()