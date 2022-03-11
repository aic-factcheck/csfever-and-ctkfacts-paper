import numpy as np

class SampleParagraphProducer:
    def __init__(self, db, seed, exclude=[], titlepar0=True):
        self.rng = np.random.RandomState(seed)
        self.db = db
        self.titlepar0 = titlepar0

        self.nrows = len(db)
        
    def __getitem__(self, id_):
        res = self.db[id_]
        if self.titlepar0:
            did = res["did"]
            row = self.db.did2blocks[did][0]
            res["title"] = self.db.txts[row]
        return res
        
    def sample(self):
        return self[self.sampleid()]

    def sampleid(self):
        m = self.rng.randint(0, self.nrows)
        return self.db.ids[m]