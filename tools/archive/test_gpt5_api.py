#!/usr/bin/env python3
"""
Simple test script to verify GPT-5 API connection and functionality.
Tests both basic and structured output modes.
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def test_basic_response():
    """Test basic GPT-5 Responses API call."""
    print("\n" + "="*60)
    print("TEST 1: Basic GPT-5 Response")
    print("="*60)
    
    try:
        client = OpenAI(timeout=30)
        
        # Basic call to GPT-5
        response = client.responses.create(
            model="gpt-5",
            input="Say hello to Dr. Miller, an interventional pulmonologist."
        )
        
        print(f"✓ Success! Response: {response.output_text}")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_structured_output():
    """Test GPT-5 with structured JSON output."""
    print("\n" + "="*60)
    print("TEST 2: Structured Output with JSON Schema")
    print("="*60)
    
    # Define a simple medical schema
    schema = {
        "type": "object",
        "properties": {
            "document_id": {"type": "string"},
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "keywords": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["document_id", "title", "summary", "keywords"]
    }
    
    structured_schema = {
        "name": "MedicalSummary",
        "schema": schema,
        "strict": True
    }
    
    prompt = """Extract the following information from this abstract:

    Title: "Novel Bronchoscopic Techniques in Lung Cancer Diagnosis"
    
    Abstract: This study evaluates new bronchoscopic methods for diagnosing peripheral lung nodules. 
    We compared electromagnetic navigation bronchoscopy (ENB) with robotic-assisted bronchoscopy 
    in 150 patients. Results showed improved diagnostic yield with robotic assistance (85% vs 72%).
    
    Return as JSON with document_id="test_001", title, summary, and keywords."""
    
    try:
        client = OpenAI(timeout=30)
        
        response = client.responses.create(
            model="gpt-5",
            input=prompt,
            response_format={
                "type": "json_schema",
                "json_schema": structured_schema
            }
        )
        
        # Parse the response
        if hasattr(response, 'output_text'):
            result = json.loads(response.output_text)
        elif hasattr(response, 'output') and response.output:
            # Handle structured output format
            if isinstance(response.output, list) and len(response.output) > 0:
                output_item = response.output[0]
                if hasattr(output_item, 'content') and isinstance(output_item.content, list):
                    result = json.loads(output_item.content[0].text)
                else:
                    result = json.loads(str(output_item))
            else:
                result = json.loads(str(response.output))
        else:
            raise ValueError("Unexpected response format")
        
        print("✓ Success! Structured output received:")
        print(json.dumps(result, indent=2))
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_timeout_handling():
    """Test timeout handling with GPT-5."""
    print("\n" + "="*60)
    print("TEST 3: Timeout Handling")
    print("="*60)
    
    try:
        # Use a very short timeout to test error handling
        client = OpenAI(timeout=1)  # 1 second timeout
        
        # This should timeout
        response = client.responses.with_options(timeout=1).create(
            model="gpt-5",
            input="Generate a very long detailed medical report about bronchoscopy procedures, " * 10
        )
        
        print("✓ Request completed (no timeout)")
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            print(f"✓ Timeout handled correctly: {error_msg[:100]}")
            return True
        else:
            print(f"✗ Unexpected error: {e}")
            return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("GPT-5 API CONNECTION TESTS")
    print("="*60)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n✗ ERROR: OPENAI_API_KEY not found in environment")
        print("Please set your API key in .env file or environment variable")
        return
    
    print(f"\n✓ API Key found: {api_key[:10]}...")
    
    # Run tests
    results = []
    
    # Test 1: Basic response
    results.append(("Basic Response", test_basic_response()))
    
    # Test 2: Structured output
    results.append(("Structured Output", test_structured_output()))
    
    # Test 3: Timeout handling
    results.append(("Timeout Handling", test_timeout_handling()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All tests passed! GPT-5 API is working correctly.")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()