import os

# A thin compatibility wrapper around the OpenAI Python SDK.
# Works with both the new SDK (>=1.x, `from openai import OpenAI`) and the legacy SDK (<=0.28).
#
# Usage:
#   from utils.openai_client import chat_complete
#   text = chat_complete(model="gpt-5-mini", messages=[...], temperature=0.2, max_tokens=900)
#
_client = None
_new_sdk = False

def _init():
    global _client, _new_sdk
    api_key = os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    try:
        # Attempt new SDK first
        from openai import OpenAI  # type: ignore
        _client = OpenAI(api_key=api_key)
        _new_sdk = True
    except Exception:
        # Fallback to legacy SDK
        import openai as openai_legacy  # type: ignore
        openai_legacy.api_key = api_key
        _client = openai_legacy
        _new_sdk = False

def chat_complete(*, model: str, messages, **kwargs) -> str:
    """Return text content from a chat completion across SDK versions."""
    global _client, _new_sdk
    if _client is None:
        _init()

    # Remove kwargs that may not exist in both SDKs
    kwargs = dict(kwargs)
    kwargs.pop("timeout", None)
    kwargs.pop("request_timeout", None)
    
    # Handle GPT-5/o1 models that use max_completion_tokens instead of max_tokens
    if _new_sdk and ("gpt-5" in model or "o1" in model):
        if "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

    if _new_sdk:
        # New SDK - no fallback, use GPT-5 models only
        resp = _client.chat.completions.create(model=model, messages=messages, **kwargs)
        return resp.choices[0].message.content or ""
    else:
        # Legacy SDK
        resp = _client.ChatCompletion.create(model=model, messages=messages, **kwargs)
        choice = resp["choices"][0]["message"]
        return choice.get("content") or ""
