import numpy as np
import os
from os.path import join as pjoin
import torch

import logging
logger = logging.getLogger(__name__)

def load_embeddings(loc):
    if os.path.isfile(loc):
        return torch.load(loc).numpy()
    else: # dir
        files = [pjoin(loc, f) for f in sorted(os.listdir(loc))]
        logger.info("loading {} embedding files from {}".format(len(files), loc))
        return np.vstack([torch.load(f).numpy() for f in files])

class SentenceEmbeddingReader:
    # TODO probably make iterator out of this
    def __init__(self, sentence_embedding_dir):
        ptfiles = [pjoin(sentence_embedding_dir, f) for f in sorted(os.listdir(sentence_embedding_dir))]
        assert len(ptfiles) > 0
        self.ptfiles = ptfiles
        self.nextid = 0
        self.opennext()
        self.dim = self.curfile.shape[1]
        
    
    def opennext(self):
        assert self.nextid < len(self.ptfiles), "only {} remains in the last file".format(self.curfile.shape[0] - self.curoff)
        fname = self.ptfiles[self.nextid]
        self.curfile = torch.load(fname)
        logger.info("opened sentence embedding PT file: {} shape: {}".format(fname, self.curfile.shape))
        self.curoff = 0
        self.nextid += 1

    def next(self, n):
        embs = torch.empty(n, self.dim, dtype=self.curfile.dtype)
        off = 0
        copied = 0
        while copied < n:
            leftinfile = self.curfile.shape[0] - self.curoff
            if leftinfile == 0:
                if self.nextid >= len(self.ptfiles):
                    return embs[0:copied+1, :]
                self.opennext()
                leftinfile = self.curfile.shape[0]
            take = min(n, leftinfile)
            embs[off:off+take] = self.curfile[self.curoff:self.curoff+take]
            n -= take
            off += take
            self.curoff += take
        return embs