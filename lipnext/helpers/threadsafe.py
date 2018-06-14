# Theadsafe iterator wrapper
# Source: https://gist.github.com/platdrag/e755f3947552804c42633a99ffd325d4

import threading


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def get_list_safe(l, index, size):
    ret = l[index:index+size]
    while size - len(ret) > 0:
        ret += l[0:size - len(ret)]
    return ret
