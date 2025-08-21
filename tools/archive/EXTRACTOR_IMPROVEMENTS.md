# Extractor GPT-5 Improvements

## Issues Fixed

The `extractor_gpt5.py` script was hanging after showing "[INFO] Using model: gpt-5" and "[INFO] Max output tokens: 8192". The following improvements have been implemented:

### 1. Empty/Invalid last_model_output.json Handling
- **Issue**: Empty or corrupted `last_model_output.json` files were not being cleaned up
- **Fix**: Added `clean_last_model_output()` function that detects and removes empty/invalid JSON files
- **Benefit**: Prevents issues with leftover corrupted state from previous failed runs

### 2. Timeout and Better Error Handling
- **Issue**: No timeout on OpenAI API calls, leading to indefinite hangs
- **Fix**: 
  - Added configurable timeout via `OPENAI_TIMEOUT` environment variable (default: 300 seconds)
  - Added proper timeout parameter to OpenAI client initialization
  - Enhanced error messages with more context
- **Benefit**: Script will fail gracefully instead of hanging indefinitely

### 3. Debug Logging
- **Issue**: No visibility into where the script was failing
- **Fix**: Added comprehensive debug logging:
  - `[DEBUG]` prefixed messages showing API call start/completion
  - Timing information for API calls
  - Character counts from API responses
  - Progress indicators for each extraction step
- **Benefit**: Easy to identify exactly where issues occur

### 4. Test Mode
- **Issue**: Required OpenAI API access to debug basic functionality
- **Fix**: Added comprehensive test mode:
  - `TEST_MODE=true` environment variable or `--test` command line flag
  - Creates realistic mock responses that validate against the schema
  - Can run without OpenAI package installed
  - Still writes to `last_model_output.json` for consistency
- **Benefit**: Can debug and test without API access or costs

### 5. API Connection Testing
- **Issue**: No way to test if API key/connection was valid
- **Fix**: Added `--test-connection` flag that:
  - Tests API connectivity with a minimal call
  - Uses cheaper gpt-3.5-turbo model for testing
  - Short 10-second timeout for quick feedback
  - Detailed error reporting
- **Benefit**: Quick way to verify API setup before running expensive extraction

## Usage Examples

### Basic Usage with Improvements
```bash
# Normal operation (with all improvements active)
python3 tools/extractor_gpt5.py --single path/to/file.json --pdf path/to/file.pdf

# Test mode (no API calls, creates mock responses)
python3 tools/extractor_gpt5.py --single path/to/file.json --test

# Test API connection
python3 tools/extractor_gpt5.py --test-connection

# With custom timeout (in seconds)
OPENAI_TIMEOUT=60 python3 tools/extractor_gpt5.py --single path/to/file.json
```

### Environment Variables
```bash
# Enable test mode
export TEST_MODE=true

# Set custom timeout (seconds)
export OPENAI_TIMEOUT=120

# Use different model
export OPENAI_MODEL=gpt-4

# Set max output tokens
export MAX_OUTPUT_TOKENS=4096
```

### Debugging a Hanging Issue
1. **First**: Test API connection
   ```bash
   python3 tools/extractor_gpt5.py --test-connection
   ```

2. **If API connection fails**: Check your API key in `.env` file

3. **If API connection works**: Try with test mode first
   ```bash
   python3 tools/extractor_gpt5.py --single your_file.json --test
   ```

4. **If test mode works**: Try with real API but shorter timeout
   ```bash
   OPENAI_TIMEOUT=60 python3 tools/extractor_gpt5.py --single your_file.json
   ```

5. **Monitor logs**: Look for debug messages to see exactly where it hangs

## New Command Line Options

- `--test`: Run in test mode (no API calls)
- `--test-connection`: Test API connection only and exit

## New Environment Variables

- `TEST_MODE`: Set to "true", "1", or "yes" to enable test mode
- `OPENAI_TIMEOUT`: Timeout in seconds for API calls (default: 300)

## Files Created/Modified

### Modified Files
- `/tools/extractor_gpt5.py`: Main script with all improvements

### New Files
- `/tools/test_extractor.py`: Comprehensive test suite for validating improvements
- `/EXTRACTOR_IMPROVEMENTS.md`: This documentation file

## Technical Implementation Details

### Timeout Implementation
- Uses `timeout` parameter in OpenAI client initialization
- Applies to all API calls (json_schema, json_object, and fallback attempts)
- Configurable via environment variable with sensible default

### Test Mode Implementation
- Checks TEST_MODE early in import process to avoid OpenAI dependency
- Creates schema-compliant mock responses
- Simulates processing time with `time.sleep(1)`
- Maintains same file output structure as real mode

### Debug Logging Strategy
- Uses `[DEBUG]` prefix for detailed internal information
- Uses `[INFO]` prefix for user-relevant information
- Includes timing information for performance monitoring
- Shows character counts to help identify empty responses

### Error Handling Improvements
- Graceful handling of missing OpenAI package in test mode
- Better error messages with actionable suggestions
- Preservation of previous error context in retry scenarios
- Clean separation between different types of failures

## Testing

Run the comprehensive test suite:
```bash
python3 tools/test_extractor.py
```

This tests:
- API connection functionality
- Test mode operation
- Debug logging output
- Timeout behavior (simulated)

## Backwards Compatibility

All changes are backwards compatible:
- Existing command line arguments work unchanged
- Existing environment variables work unchanged
- Default behavior is unchanged when no new options are used
- Output format and file locations remain the same