#!/usr/bin/env python3
"""
Test GPT-5 availability and create a working configuration
"""

import os
from dotenv import load_dotenv
import openai
import json

load_dotenv()

def test_model_configurations():
    """Test different model and parameter combinations"""
    
    client = openai.OpenAI()
    
    test_configs = [
        # GPT-5 configurations
        {"model": "gpt-5", "params": {"max_completion_tokens": 100}},
        {"model": "gpt-5", "params": {"max_tokens": 100}},
        {"model": "gpt-5-mini", "params": {"max_completion_tokens": 100}},
        {"model": "gpt-5-mini", "params": {"max_tokens": 100}},
        
        # O1 model configurations
        {"model": "o1-preview", "params": {"max_completion_tokens": 100}},
        {"model": "o1-mini", "params": {"max_completion_tokens": 100}},
        
        # GPT-4 configurations (fallback)
        {"model": "gpt-4o", "params": {"max_tokens": 100, "temperature": 0.2}},
        {"model": "gpt-4o-mini", "params": {"max_tokens": 100, "temperature": 0.2}},
        {"model": "gpt-4-turbo-preview", "params": {"max_tokens": 100, "temperature": 0.2}},
    ]
    
    working_configs = []
    
    print("=" * 60)
    print("Testing Model Configurations")
    print("=" * 60)
    
    for config in test_configs:
        model = config["model"]
        params = config["params"]
        
        print(f"\nTesting: {model}")
        print(f"Parameters: {params}")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "What is EBUS? Answer in one sentence."}
                ],
                **params
            )
            
            print(f"✅ SUCCESS!")
            print(f"   Response: {response.choices[0].message.content[:100]}")
            
            working_configs.append({
                "model": model,
                "params": params,
                "response_model": response.model
            })
            
        except Exception as e:
            error_msg = str(e)
            if "does not exist" in error_msg or "model_not_found" in error_msg:
                print(f"❌ Model not available")
            elif "max_tokens" in error_msg and "max_completion_tokens" in error_msg:
                print(f"❌ Wrong token parameter")
            elif "temperature" in error_msg:
                print(f"❌ Temperature not supported")
            else:
                print(f"❌ Error: {error_msg[:100]}")
    
    print("\n" + "=" * 60)
    print("WORKING CONFIGURATIONS:")
    print("=" * 60)
    
    if working_configs:
        for config in working_configs:
            print(f"\n✅ Model: {config['model']}")
            print(f"   Parameters: {config['params']}")
            print(f"   Actual model used: {config['response_model']}")
        
        # Save working config
        with open("working_model_config.json", "w") as f:
            json.dump(working_configs, f, indent=2)
        print(f"\n💾 Working configurations saved to working_model_config.json")
        
        # Recommend best models
        print("\n" + "=" * 60)
        print("RECOMMENDED CONFIGURATION:")
        print("=" * 60)
        
        # Find best quick model
        quick_priority = ["gpt-5-mini", "o1-mini", "gpt-4o-mini", "gpt-4-turbo-preview"]
        quick_model = None
        for model_name in quick_priority:
            for config in working_configs:
                if config["model"] == model_name:
                    quick_model = config
                    break
            if quick_model:
                break
        
        # Find best depth model
        depth_priority = ["gpt-5", "o1-preview", "gpt-4o", "gpt-4-turbo-preview"]
        depth_model = None
        for model_name in depth_priority:
            for config in working_configs:
                if config["model"] == model_name:
                    depth_model = config
                    break
            if depth_model:
                break
        
        if quick_model:
            print(f"\n🚀 Quick Mode: {quick_model['model']}")
            print(f"   Parameters: {quick_model['params']}")
        
        if depth_model:
            print(f"\n🔬 Depth Mode: {depth_model['model']}")
            print(f"   Parameters: {depth_model['params']}")
        
        # Create fixed config file
        recommended_config = {
            "quick_mode": quick_model,
            "depth_mode": depth_model
        }
        
        with open("recommended_model_config.json", "w") as f:
            json.dump(recommended_config, f, indent=2)
        
        print(f"\n💾 Recommended configuration saved to recommended_model_config.json")
        
    else:
        print("\n❌ No working configurations found!")
        print("Please check your API key and permissions.")
    
    return working_configs

if __name__ == "__main__":
    configs = test_model_configurations()
    
    if configs:
        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("=" * 60)
        print("1. Review the recommended_model_config.json file")
        print("2. Update bronchmonkey_pro.py with the working models")
        print("3. Use the exact parameters that worked")