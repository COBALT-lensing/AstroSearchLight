import functools
import hashlib
from pathlib import Path
from pickle import UnpicklingError

import compress_pickle as pickle


FUNC_CACHE_PATH = "data/cache/"


def cache(func):
    """Decorator for caching function results to disk."""

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

        p_hash = hashlib.md5(repr(tuple(hash_parts)).encode()).hexdigest()
        cache_file = (
            Path(FUNC_CACHE_PATH)
            / func.__name__
            / p_hash[:2]
            / p_hash[2:4]
            / f"{p_hash}.pkl.bz2"  # bz2 seems to be the best for RGB data
        )

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except (UnpicklingError, EOFError, ValueError):
                pass

        result = func(*args, **kwargs)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)

        return result

    return _cache_f
