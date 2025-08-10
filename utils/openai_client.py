import os
import json
from pathlib import Path

_client = None
_new_sdk = False

def _init():
    global _client, _new_sdk
    api_key = os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI  # >=1.x
        _client = OpenAI(api_key=api_key)
        _new_sdk = True
    except Exception:
        import openai as openai_legacy  # <=0.28.x
        openai_legacy.api_key = api_key
        _client = openai_legacy
        _new_sdk = False

def chat_complete_text(**kwargs):
    """Return (text, info). Try Chat Completions, then fallback to Responses API. Never return empty string."""
    global _client, _new_sdk
    if _client is None:
        _init()

    info = {"route": None, "error": None, "raw": None}
    text = ""

    try:
        info["route"] = "chat.completions"
        if _new_sdk:
            resp = _client.chat.completions.create(**kwargs)
            info["raw"] = resp.to_dict()
            text = _extract_text(resp.choices[0].message.content)
        else:
            resp = _client.ChatCompletion.create(**kwargs)
            info["raw"] = resp
            text = _extract_text(resp["choices"][0]["message"]["content"])
    except Exception as e:
        info["error"] = str(e)

    if not text:
        try:
            info["route"] = "responses.create"
            if _new_sdk:
                resp = _client.responses.create(model=kwargs["model"], input=kwargs["messages"])
                info["raw"] = resp.to_dict()
                text = _extract_text(getattr(resp, "output_text", None))
        except Exception as e:
            info["error"] = str(e)

    Path("data").mkdir(exist_ok=True)
    Path("data/last_llm_debug.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    if not text:
        text = "⚠️ I couldn't generate an answer. See data/last_llm_debug.json for details."
    return text, info

def _extract_text(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(str(c) for c in content if c).strip()
    return ""
