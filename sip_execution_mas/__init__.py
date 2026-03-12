"""
sip-execution-mas — 6-Node Gemini-Powered SIP Execution MAS
=============================================================
SSL is patched here (first thing) so yfinance/httpx/requests all
pick up the certifi CA bundle on macOS python.org installs.
"""

# ── SSL fix — must run before any yfinance / requests / httpx import ──────────
import os as _os
import ssl as _ssl

try:
    import certifi as _certifi
    _cafile = _certifi.where()
    _os.environ["SSL_CERT_FILE"]      = _cafile
    _os.environ["REQUESTS_CA_BUNDLE"] = _cafile
    _os.environ["CURL_CA_BUNDLE"]     = _cafile

    _orig_ctx = _ssl.create_default_context

    def _patched_ctx(*args, **kwargs):
        ctx = _orig_ctx(*args, **kwargs)
        try:
            ctx.load_verify_locations(cafile=_cafile)
        except Exception:
            pass
        return ctx

    _ssl.create_default_context = _patched_ctx
except ImportError:
    pass
