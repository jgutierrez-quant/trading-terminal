# Apply urllib3 v2 compat patch for pytrends before any sentiment imports.
# pytrends 4.9.2 passes `method_whitelist` to urllib3.util.retry.Retry,
# but urllib3 v2 renamed that arg to `allowed_methods`.
import urllib3.util.retry as _retry_mod

_orig_retry_init = _retry_mod.Retry.__init__

def _patched_retry_init(self, *args, **kwargs):
    if "method_whitelist" in kwargs:
        kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
    _orig_retry_init(self, *args, **kwargs)

_retry_mod.Retry.__init__ = _patched_retry_init
