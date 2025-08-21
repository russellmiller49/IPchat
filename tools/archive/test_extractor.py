#!/usr/bin/env python3
"""
Test script for extractor_gpt5.py improvements
Tests timeout, error handling, test mode, and API connection
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def test_connection():
    """Test the API connection functionality."""
    print("=== Testing API Connection ===")
    result = subprocess.run([
        sys.executable, "tools/extractor_gpt5.py", "--test-connection"
    ], capture_output=True, text=True)
    
    print(f"Return code: {result.returncode}")
    print(f"STDOUT: {result.stdout}")
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    return result.returncode == 0

def test_test_mode():
    """Test the test mode functionality."""
    print("\n=== Testing Test Mode ===")
    
    # Create a minimal test JSON file
    test_json = Path("test_sample.json")
    test_json.write_text("""
    {
        "document_id": "test_doc_123",
        "title": "Test Medical Paper",
        "content": "This is a test medical paper about interventional pulmonology."
    }
    """)
    
    try:
        result = subprocess.run([
            sys.executable, "tools/extractor_gpt5.py", 
            "--single", str(test_json), "--test"
        ], capture_output=True, text=True, timeout=30)
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
            
        # Check if output file was created
        output_file = Path("data/outputs/test_sample.structured.json")
        if output_file.exists():
            print("✓ Output file created successfully")
            content = output_file.read_text()
            print(f"Output file size: {len(content)} characters")
            # Check if it's valid JSON
            import json
            try:
                data = json.loads(content)
                print("✓ Output is valid JSON")
                if "source" in data and "document_id" in data["source"]:
                    print("✓ Output contains expected structure")
                else:
                    print("⚠ Output missing expected fields")
            except json.JSONDecodeError as e:
                print(f"✗ Output is not valid JSON: {e}")
        else:
            print("✗ No output file created")
            
        return result.returncode == 0
        
    finally:
        # Clean up
        if test_json.exists():
            test_json.unlink()
        output_file = Path("data/outputs/test_sample.structured.json")
        if output_file.exists():
            output_file.unlink()

def test_timeout_handling():
    """Test timeout handling by setting a very short timeout."""
    print("\n=== Testing Timeout Handling ===")
    
    # Set a very short timeout
    env = os.environ.copy()
    env["OPENAI_TIMEOUT"] = "1"  # 1 second timeout
    
    test_json = Path("test_timeout.json")
    test_json.write_text("""
    {
        "document_id": "timeout_test",
        "content": "This should timeout quickly due to the 1-second timeout setting."
    }
    """)
    
    try:
        start_time = time.time()
        result = subprocess.run([
            sys.executable, "tools/extractor_gpt5.py", 
            "--single", str(test_json)
        ], capture_output=True, text=True, timeout=10, env=env)
        elapsed = time.time() - start_time
        
        print(f"Process completed in {elapsed:.1f} seconds")
        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
            
        # Should fail due to timeout, but gracefully
        if result.returncode != 0 and "timeout" in result.stdout.lower():
            print("✓ Timeout handled gracefully")
            return True
        else:
            print("⚠ Timeout not detected or not handled as expected")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Process hung despite timeout setting")
        return False
    finally:
        if test_json.exists():
            test_json.unlink()

def test_debug_logging():
    """Test that debug logging is working."""
    print("\n=== Testing Debug Logging ===")
    
    test_json = Path("test_debug.json")
    test_json.write_text("""
    {
        "document_id": "debug_test",
        "content": "Testing debug logging functionality."
    }
    """)
    
    try:
        result = subprocess.run([
            sys.executable, "tools/extractor_gpt5.py", 
            "--single", str(test_json), "--test"
        ], capture_output=True, text=True, timeout=30)
        
        output = result.stdout
        debug_markers = [
            "[DEBUG]",
            "Starting extraction process",
            "Creating mock response for test mode"
        ]
        
        found_markers = [marker for marker in debug_markers if marker in output]
        print(f"Found debug markers: {found_markers}")
        
        if len(found_markers) >= 2:
            print("✓ Debug logging is working")
            return True
        else:
            print("⚠ Debug logging may not be fully working")
            return False
            
    finally:
        if test_json.exists():
            test_json.unlink()
        output_file = Path("data/outputs/test_debug.structured.json")
        if output_file.exists():
            output_file.unlink()

def main():
    """Run all tests."""
    print("Testing extractor_gpt5.py improvements...\n")
    
    results = {}
    
    # Test 1: API Connection (may fail if no valid key, but shouldn't hang)
    try:
        results["connection"] = test_connection()
    except Exception as e:
        print(f"Connection test failed with exception: {e}")
        results["connection"] = False
    
    # Test 2: Test Mode (should always work)
    try:
        results["test_mode"] = test_test_mode()
    except Exception as e:
        print(f"Test mode failed with exception: {e}")
        results["test_mode"] = False
    
    # Test 3: Debug Logging (should always work)
    try:
        results["debug_logging"] = test_debug_logging()
    except Exception as e:
        print(f"Debug logging test failed with exception: {e}")
        results["debug_logging"] = False
    
    # Test 4: Timeout Handling (may be tricky to test reliably)
    try:
        results["timeout"] = test_timeout_handling()
    except Exception as e:
        print(f"Timeout test failed with exception: {e}")
        results["timeout"] = False
    
    # Summary
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:20} {status}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    print(f"\nOverall: {passed_count}/{total_count} tests passed")
    
    if passed_count >= 2:  # At least test_mode and debug_logging should work
        print("✓ Core improvements are working")
        return True
    else:
        print("✗ Some core improvements may not be working")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)