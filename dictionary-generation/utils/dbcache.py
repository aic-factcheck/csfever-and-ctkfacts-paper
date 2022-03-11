from collections import defaultdict, OrderedDict
import datetime
import numpy as np
import sqlite3
from tqdm import tqdm
import unicodedata as ud
import sys

import logging
logger = logging.getLogger(__name__)

from utils.tokenization import detokenize, detokenize2

class DBCache(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id) as in memory db with hash indices.
    
    Based on DRQA DocDB class.
    """

    def __init__(self, db_path, excludekw=""):
        self.path = db_path
        excludekw = [e.lower() for e in excludekw.split(";")]
        with sqlite3.connect(self.path, detect_types=sqlite3.PARSE_DECLTYPES) as connection:
            logger.info(f"reading database to RAM")
            logger.info(f"excluding keywords: {excludekw}")

            def hasexcludedkw(kws):
                for k in kws:
                    for e in excludekw:
                        if e.startswith(k):
                            return True
                return False

            nrows = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            logger.info(f"processing total {nrows} rows")

            cursor = connection.cursor()
            cursor.execute("SELECT id, did, bid, date, keywords, text FROM documents")
            self.txts = []
            self.kws = []
            self.dates = []
            self.id2row = {}
            self.ids = []
            self.dids = []
            self.bids = []
            self.titles = set()
            self.bodies = set()
            self.did2blocks = defaultdict(list) # maps dids to row numbers

            row = 0
            nexcluded = 0
            for id_, did, bid, date_, keywords, text in tqdm(cursor.fetchall(), total=nrows, mininterval=10.0):
                keywords = [k.lower() for k in keywords.split(";")]
                if hasexcludedkw(keywords):
                    nexcluded += 1
                    continue
                self.txts.append(text)
                self.kws.append(keywords)
                self.dates.append(datetime.datetime.utcfromtimestamp(date_ * 1e-9))
                self.id2row[id_] = row
                self.ids.append(id_)
                self.dids.append(did)
                self.bids.append(bid)
                if bid == 0:
                    self.titles.add(row)
                else:
                    self.bodies.add(row)
                self.did2blocks[did].append(row) # expects that the blocks are sorted in DBs
                row += 1
            cursor.close()
            logger.info(f"blocks imported: {row}, excluded based on keywords: {nexcluded}")                      

    def get_block_ids(self):
        return self.id2row.keys()

    def get_document_ids(self):
        return self.did2blocks.keys()

    def get_block_text(self, id_):
        return self.txts[self.id2row[ud.normalize("NFD", id_)]]

    def get_block_texts(self, did, f=lambda txt: txt, title=True):
        return OrderedDict((self.ids[i], f(self.txts[i])) for i in self.did2blocks[did] if title or (self.bids[i] != 0))

    def get_document(self, did, f=detokenize2, title=True, block_join="\n\n"):
        blocks = [f(self.txts[i]) for i in self.did2blocks[did] if title or (self.bids[i] != 0)]
        if block_join is None:
            return blocks
        return block_join.join(blocks)

    def hasid(self, id_):
        return id_ in self.id2row

    def id2did(self, id_):
        return self.dids[self.id2row[id_]]

    def id2bid(self, id_):
        return self.bids[self.id2row[id_]]

    def id2date(self, id_):
        return self.dates[self.id2row[id_]]

    def did2ids(self, did):
        return [self.ids[i] for i in self.did2blocks[did]]

    def did2date(self, did):
        return self.dates[self.did2blocks[did][0]]
    
    def istitle(self, id_):
        return self.hasid(id_) and self.id2bid(id_) == 0
    
    def __len__(self):
        return len(self.id2row)
    
    def __getitem__(self, id_, detokenize=detokenize2):
        row = self.id2row[id_]
        res = {"id": id_, "did": self.dids[row], "bid": self.bids[row], "date": self.dates[row], "keywords": self.kws[row], "text": detokenize(self.txts[row])}
        return res