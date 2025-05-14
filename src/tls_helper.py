# tls_helper.py
import threading
from iterdecon_cython import ThreadLocalStorage

_tls_store = threading.local()

def get_tls(nsamp: int) -> ThreadLocalStorage:
    """Get or create a thread-local storage object matching the given nsamp."""
    tls = getattr(_tls_store, "tls", None)

    if tls is not None and tls.nfft == nsamp:
        return tls

    if tls is not None:
        tls.cleanup()  # Safe cleanup if nfft mismatched

    _tls_store.tls = ThreadLocalStorage(nsamp)
    return _tls_store.tls
