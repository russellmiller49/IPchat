import os
import streamlit as st

def parse_basic_auth_env(env_var: str = "BASIC_AUTH_USERS"):
    """Parse env like: 'alice:pw1,bob:pw2' -> [('alice','pw1'), ('bob','pw2')]"""
    raw = os.getenv(env_var, "").strip()
    pairs = []
    for item in raw.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        u, p = item.split(":", 1)
        pairs.append((u.strip(), p.strip()))
    return pairs

def check_password():
    """Simple username/password gate using BASIC_AUTH_USERS env var. Returns True when authenticated."""
    allowed = parse_basic_auth_env()
    if not allowed:
        # No credentials set -> allow (useful for local dev)
        return True

    def _verify():
        user = st.session_state.get("username", "")
        pw = st.session_state.get("password", "")
        st.session_state["password_ok"] = (user, pw) in allowed
        # don't retain password
        if "password" in st.session_state:
            del st.session_state["password"]

    if "password_ok" not in st.session_state:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password", on_change=_verify)
        st.stop()

    if not st.session_state["password_ok"]:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password", on_change=_verify)
        st.error("Access denied. Check username & password.")
        st.stop()

    return True