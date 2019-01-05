import time


def logfunc(func):
    "Decorator that logs the geven function's runnng time"

    def logged(*args, **kwargs):
        print('===> running', func.__name__, '...')
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print('<=== finished {} in {:03.2f} s.'.format(func.__name__, elapsed))
        return result
    return logged
