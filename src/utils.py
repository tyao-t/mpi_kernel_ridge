import gc
from functools import wraps

def auto_gc(enabled=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if enabled:
                gc.collect()
            return result
        return wrapper
    return decorator
