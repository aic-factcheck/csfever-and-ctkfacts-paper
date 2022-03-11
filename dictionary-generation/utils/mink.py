import numpy as np

class MinKStorage:
    def __init__(self, k):
        # TODO better use heap for this
        self.k = k
        self._keys = np.full(k, -1, dtype=np.int)
        self._vals = np.full(k, np.inf, dtype=np.float32)
        self.n = 0
        
    def add(self, key, val):
        if self.n < self.k or val < self._vals[0]:
            idx = np.searchsorted(self._vals[:self.n], val)
            self._keys[idx+1:] = self._keys[idx:-1]
            self._keys[idx] = key
            self._vals[idx+1:] = self._vals[idx:-1]
            self._vals[idx] = val
            self.n += 1
            
    def keys(self):
        return self._keys[0:min(self.k, self.n)]

    def vals(self):
        return self._vals[0:min(self.k, self.n)]
