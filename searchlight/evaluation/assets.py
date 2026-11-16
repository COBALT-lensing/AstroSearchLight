"""Physical-system identities for basic independence checks."""
import json
import numpy as np


def physical_identity(params):
    """Conservative binary identity, ignoring noise, event epoch, and distance."""
    keys = ('a', 'i', 'Macc', 'Mcomp', 'Rcomp', 'e', 'periapsis_phase')
    try:
        values = [float(params[k]) for k in keys]
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(values).all():
        return None
    import hashlib
    normalized = [format(value, '.12g') for value in values]
    return 'physical:'+hashlib.sha256(json.dumps(normalized).encode()).hexdigest()

