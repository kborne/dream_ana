from typing import Optional, List, Any
import importlib

def deep_merge(orig, new):
    """
    Recursively merge `new` into `orig`. Both `orig` and `new` must be dicts.
    """
    for key, val in new.items():
        existing = orig.get(key)
        if isinstance(existing, dict) and isinstance(val, dict):
            deep_merge(existing, val)
        else:
            orig[key] = val

def head_match(a: str, b: List[str]) -> bool:
    """
    Return True if `a` is a prefix (“head”) of any element in `b`.
    Uses str.startswith (C‐level) and short‐circuits on first match.
    """
    # localize for speed
    startswith = str.startswith
    for s in b:
        if startswith(s, a):
            return True
    return False

def lists_intersection(a: List[Any], b: List[Any]) -> List[Any]:
    """
    Return a list of the unique elements that appear in both a and b,
    in the order they first appear in a.
    """
    b_set = set(b)
    seen = set()
    c: List[Any] = []
    for x in a:
        if x in b_set and x not in seen:
            seen.add(x)
            c.append(x)
    return c


# helper for dynamic or identity function
def mk_func(func_name: Optional[str]):
    if func_name is None:
        return lambda arr, *args, **kwargs: arr
    module_name, fn = func_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, fn)