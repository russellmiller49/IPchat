#!/usr/bin/env python3
"""
Test available OpenAI models and their parameters
"""

import os
from dotenv import load_dotenv
import openai

load_dotenv()

def test_models():
    """Test which models are available and their parameter requirements"""
    
    client = openai.OpenAI()
    
    # Models to test based on the codebase references
    test_models = [
        ("gpt-4o-mini", "GPT-4 Optimized Mini"),
        ("gpt-4o", "GPT-4 Optimized"),
        ("gpt-5-mini", "GPT-5 Mini"),
        ("gpt-5", "GPT-5"),
        ("gpt-5-2025-08-07", "GPT-5 August 2025"),
        ("o1-mini", "O1 Mini (Potential GPT-5)"),
        ("o1-preview", "O1 Preview (Potential GPT-5)"),
    ]
    
    print("Testing OpenAI Model Availability")
    print("="*60)
    
    for model_name, description in test_models:
        print(f"\nTesting: {model_name} ({description})")
        print("-"*40)
        
        try:
            # Try with max_tokens (traditional parameter)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": "Say 'test' in one word"}
                ],
                max_tokens=10,
                temperature=0
            )
            print(f"✅ Works with max_tokens")
            print(f"   Response: {response.choices[0].message.content}")
            
        except Exception as e:
            error_str = str(e)
            if "max_completion_tokens" in error_str:
                # Try with max_completion_tokens
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "user", "content": "Say 'test' in one word"}
                        ],
                        max_completion_tokens=10,
                        temperature=0
                    )
                    print(f"✅ Works with max_completion_tokens (GPT-5 parameter)")
                    print(f"   Response: {response.choices[0].message.content}")
                except Exception as e2:
                    print(f"❌ Failed: {str(e2)[:100]}")
            elif "does not exist" in error_str or "model_not_found" in error_str:
                print(f"❌ Model not available")
            else:
                print(f"❌ Error: {error_str[:100]}")
    
    print("\n" + "="*60)
    print("Recommendations based on test results:")
    print("- Use models that show ✅ above")
    print("- For GPT-5 models, use 'max_completion_tokens' parameter")
    print("- For GPT-4 models, use 'max_tokens' parameter")

if __name__ == "__main__":
    test_models()