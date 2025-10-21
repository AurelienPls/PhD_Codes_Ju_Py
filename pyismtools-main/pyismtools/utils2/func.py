import collections
import typing

class FIFOCache(collections.OrderedDict):
    def __init__(self, size = 10):
        collections.OrderedDict.__init__(self, [])
        self.size = size

    def __setitem__(self, key, value):
        collections.OrderedDict.__setitem__(self, key, value)
        if len(self) > self.size:
            self.popitem(last = False)

def makeFIFOKVCache(size = 10):

    _cache = collections.orderedDict()

    def _add(k, v):
        _cache[k] = v
        if len(_cache) > size:
            _cache.popitem(last = False)

    _protocol = {
            # exceptions are handled by the orderedDict data structure.
            'add': _add,
            'get': lambda k: _cache[k],
            'contains': lambda k: k in _cache
            }

    def _proxy(action, *args):
        if action in _protocol:
            return _protocol[action](*args) 
        else:
            raise Exception('protocol service {} is not available')

    return _proxy

class FIFOKVCacheProtocol(typing.Protocol):

    def __init__(self, size = 10):
        pass

    def _add(self, k, v):
        pass

    def get(self, k):
        pass

    def contains(self, k):
        pass

class FIFOKVCache:

    def __init__(self, size = 10):
        this._cache = collections.OrderedDict()
        self._size = size

    def add(self, k, v):
        this._cache[k] = v
        if len(self._cache) > self._size:
            self.cache.popitem(last = False)

    def get(self, k):
        return self._cache[k]

    def contains(self, k):
        return k in self._cache

def memoize(fn, size = 10):
    " size is the number of elements to keep in cache before poping the oldest"
    cache = collections.OrderedDict([])
    def _memoized(*args):
        if args not in cache:
            print("miss cache")
            cache[args] = fn(*args)
            if len(cache) > size:
                cache.popitem(last = True) # FIFO
        return cache[(args)]

    return _memoized
    
def first(coll):
    """
    returns the first element of coll.

    Raise TypeError if coll is not
    iterable and StopIteration if no more elements
    """
    
    # see alternative solutions
    # http://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable

    # here we use pythonic duck typing
    try:
        iterator = iter(coll)
    except TypeError: # not iterable
        raise
    else: # iterable
        return next(iterator)

def second(coll):
    """
    return the second element of coll.

    If coll is not iterable raises
    TypeError.
    If coll has less than 2 elms, raise StopIteration
    """
    
    try:
        iterator = iter(coll)
    except TypeError:
        raise
    else:
        next(iterator)
        return next(iterator)

def take(stream, cnt):
    """take/pop the first cnt elements of stream.

    Similar to clojure's take

    If the stream is less than cnt long, returns the list up to the
    end of the stream, i.e the returned list will be less than cnt
    long
    
    returns a list (not a stream/iterator/generator !)

    """
    
    to_return = []

    try:
        iterator = iter(stream)
        try:
            for i in range(cnt):
                to_return.append(next(iterator))
        except StopIteration:
            pass
    except TypeError:
        raise

    return to_return

def get_assoc(aa, key, default = None, all_ = False):
    r = [v for k, v in aa if k == key]
    
    if r:
        if all_ == False:
            r = r[0]
    else:
        r = default
       
    return r

ga = get_assoc

def update_assoc(a, k, v):
    na = []
    for ok, ov in a:
        if ok == k:
            na.append((ok, v))
        else:
            na.append((ok, ov))

    return na

def in_assoc(aa, key):
    try:
        return next(k for k, v in aa if key == k)
    except:
        return False
