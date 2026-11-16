import functools
import hashlib
import inspect
import logging
import os
from pathlib import Path
from pickle import PicklingError, UnpicklingError

import compress_pickle as pickle

from .paths import CACHE_PATH

FUNC_CACHE_PATH = CACHE_PATH
logger = logging.getLogger(__name__)


CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "true").lower() == "true"


def cache(func=None, *, file_args=()):
    """Cache results, optionally invalidating when named local files change."""
    if func is None:
        return lambda wrapped: cache(wrapped, file_args=file_args)
    signature = inspect.signature(func) if file_args else None

    def _is_hashable(value):
        try:
            hash(value)
            return True
        except TypeError:
            return False

    @functools.wraps(func)
    def _cache_f(*args, **kwargs):
        hash_parts = []

        for idx, value in enumerate(args):
            if _is_hashable(value):
                hash_parts.append(("arg", idx, type(value).__qualname__, repr(value)))

        for key in sorted(kwargs):
            value = kwargs[key]
            if _is_hashable(value):
                hash_parts.append(("kwarg", key, type(value).__qualname__, repr(value)))

        if file_args:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            for name in file_args:
                value = bound.arguments[name]
                # Remote MAST/HTTP curves have no local file metadata.
                if "://" in str(value) or str(value).startswith("mast:"):
                    continue
                path = Path(value).expanduser().resolve()
                stat = path.stat()
                hash_parts.append(("file", name, str(path), stat.st_size,
                                   stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino))

        p_hash = hashlib.md5(repr(tuple(hash_parts)).encode()).hexdigest()
        cache_file = (
            Path(FUNC_CACHE_PATH)
            / func.__name__
            / p_hash[:2]
            / p_hash[2:4]
            / f"{p_hash}.pkl.bz2"  # bz2 seems to be the best for RGB data
        )

        if CACHE_ENABLED and cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except (UnpicklingError, EOFError, ValueError):
                pass

        result = func(*args, **kwargs)
        if CACHE_ENABLED:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
            except (PicklingError, AttributeError, TypeError) as exc:
                # The cache should be best-effort rather than making the underlying
                # function unusable if the result can't be pickled.
                cache_file.unlink(missing_ok=True)
                logger.warning(
                    "Skipping cache write for %s at %s because pickling failed: %s",
                    func.__name__,
                    cache_file,
                    exc,
                )

        return result

    return _cache_f
