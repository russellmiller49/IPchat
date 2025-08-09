#!/usr/bin/env python3
"""
Simple test of medical evidence extraction with GPT-5.
This tests the corrected API usage.
"""

import json
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def test_simple_extraction():
    """Test a simple medical extraction using GPT-5 Responses API."""
    
    # Sample medical text
    medical_text = """
    Title: Efficacy of Robotic Bronchoscopy in Peripheral Lung Nodule Diagnosis
    
    Authors: Smith JA, Johnson RB, Williams KC
    Journal: Journal of Interventional Pulmonology, 2024
    
    Abstract:
    Background: Diagnosis of peripheral lung nodules remains challenging. This study evaluates 
    the diagnostic yield of robotic-assisted bronchoscopy (RAB) compared to conventional methods.
    
    Methods: We enrolled 250 patients with peripheral lung nodules 1-3 cm in size. Patients were 
    randomized to either RAB (n=125) or electromagnetic navigation bronchoscopy (ENB) (n=125).
    
    Results: Diagnostic yield was significantly higher with RAB (88.8%) compared to ENB (73.6%), 
    p<0.001. Procedure time was similar (45 vs 42 minutes). Complications were rare in both groups 
    (2.4% vs 3.2%, p=NS). Pneumothorax occurred in 1 patient in each group.
    
    Conclusions: Robotic bronchoscopy shows superior diagnostic yield for peripheral lung nodules 
    with similar safety profile compared to ENB.
    """
    
    # Create extraction prompt
    prompt = f"""You are a medical evidence extraction specialist. Extract key information from this medical paper.

PAPER CONTENT:
{medical_text}

Extract and return as JSON with these fields:
- title: paper title
- authors: list of author names  
- year: publication year
- study_type: type of study (RCT, observational, etc.)
- sample_size: total number of patients
- intervention: main intervention studied
- comparator: what it was compared to
- primary_outcome: main outcome measure and result
- safety: key safety findings
- conclusion: main conclusion

Return ONLY the JSON object:"""

    print("Testing GPT-5 Medical Evidence Extraction")
    print("="*60)
    
    try:
        # Initialize OpenAI client
        client = OpenAI(timeout=60)
        
        print("Calling GPT-5 API...")
        
        # Call GPT-5 using Responses API
        response = client.responses.create(
            model="gpt-5",
            input=prompt
        )
        
        # Get the response text
        if hasattr(response, 'output_text'):
            result_text = response.output_text
        else:
            print("ERROR: Unexpected response format")
            return False
        
        print("\nRaw Response:")
        print("-"*40)
        print(result_text)
        print("-"*40)
        
        # Try to parse as JSON
        try:
            # Clean up the response if needed
            if result_text.startswith("```"):
                # Remove code fences
                lines = result_text.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith("```"):
                        in_json = not in_json
                    elif in_json or (not line.startswith("```")):
                        if not line.startswith("```"):
                            json_lines.append(line)
                result_text = '\n'.join(json_lines)
            
            result = json.loads(result_text)
            
            print("\nExtracted Data:")
            print("-"*40)
            print(json.dumps(result, indent=2))
            print("-"*40)
            
            print("\n✓ SUCCESS: Extraction completed successfully!")
            return True
            
        except json.JSONDecodeError as e:
            print(f"\nWARNING: Could not parse JSON: {e}")
            print("Raw output saved to: test_extraction_output.txt")
            
            with open("test_extraction_output.txt", "w") as f:
                f.write(result_text)
            
            return False
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print("\nPossible issues:")
        print("1. Check that OPENAI_API_KEY is set correctly")
        print("2. Ensure you have access to GPT-5 model")
        print("3. Check your OpenAI account has sufficient credits")
        return False


def main():
    """Run the test."""
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found!")
        print("Please set it in your .env file or environment")
        return
    
    print(f"API Key found: {os.getenv('OPENAI_API_KEY')[:10]}...")
    print()
    
    # Run the test
    success = test_simple_extraction()
    
    if success:
        print("\n" + "="*60)
        print("TEST PASSED - GPT-5 extraction is working!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("TEST FAILED - Please check the errors above")
        print("="*60)


if __name__ == "__main__":
    main()