import time


def get_time(func):
    def middle(*args, **kwargs):
        start_time = time.time()
        returns = func(*args, **kwargs)
        duration = time.time() - start_time
        return returns, duration

    return middle
