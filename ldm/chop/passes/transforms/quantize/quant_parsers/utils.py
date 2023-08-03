from copy import deepcopy
from typing import Dict, Tuple


def cp_multi_values(src: Dict, dst: Dict, src_keys: Tuple, dst_keys: Tuple = None):
    """Copy multiple values from src dict to dst dict."""
    if dst_keys is None:
        for key in src_keys:
            dst[key] = deepcopy(src[key])
    else:
        for src_key, dst_key in zip(src_keys, dst_keys):
            dst[dst_key] = deepcopy(src[src_key])


def has_multi_keys(src: Dict, keys: Tuple):
    """Check if src dict has multiple keys."""
    for key in keys:
        if key not in src:
            return False
    return True
