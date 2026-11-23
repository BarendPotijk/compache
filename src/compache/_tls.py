"""Root certificates for a Python that ships none, as the python.org builds for macOS do.

Every HTTPS call here first goes out with Python's own certificates. When the
host cannot be verified with those, `fall_back()` switches the process to
certifi's bundle or the system's, and the call is made once more.
"""

from __future__ import annotations

import os
import ssl

__all__ = ["FOUND", "fall_back", "unverified"]

# The context HTTPS calls are made with. None stands for Python's own default.
FOUND: dict = {"context": None}
_BUNDLES = (
    "/etc/ssl/cert.pem",
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
)


def unverified(error: BaseException) -> bool:
    """Whether a call failed because the host's certificate could not be checked."""
    return "CERTIFICATE_VERIFY_FAILED" in f"{error} {getattr(error, 'reason', '')}"


def _other() -> ssl.SSLContext | None:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    for bundle in _BUNDLES:
        if os.path.exists(bundle):
            return ssl.create_default_context(cafile=bundle)
    return None


def fall_back() -> bool:
    """Switch to other root certificates. False when that was done before, or there are none."""
    if FOUND["context"] is not None:
        return False
    FOUND["context"] = _other()
    return FOUND["context"] is not None
