import json
import torch

from ner import UFALNERExtractor
from prediction.retrieval import TopNDocsDRQA, TopNDocsDRQATwoTower, TopNDocsTwoTowerFaiss
from sampling import SampleParagraphProducer
from utils.dbcache import DBCache

import logging
logger = logging.getLogger(__name__)

class State:
    def __init__(self, ner_model, db_name, kw_model, sem_model, sem_embeddings, sem_faiss_index, excludekw="", gpu=None):
        if gpu is None:
            gpu = torch.cuda.is_available()
        self.db = DBCache(db_name, excludekw=excludekw)

        self.spp = SampleParagraphProducer(self.db, seed=None)
        self.kwmodel = TopNDocsDRQA(self.db, kw_model)

        self.semmodel = TopNDocsTwoTowerFaiss(
            db_name=db_name,
            model_dir=sem_model,
            embeddings=sem_embeddings,
            gpu=gpu,
            faissindex=sem_faiss_index)
    
    # semmodel_titles = TopNDocsTwoTowerFaiss(
    #     db_name=dbname,
    #     model_dir="/mnt/data/factcheck/ict_pretrained_models/sentence-transformers/mbert_finetuned_best_ict_1.3",
    #     embeddings=f"/mnt/data/factcheck/CTK/{dbver}/emb/embedded_pages_mbert_finetuned_best_ict_1.3_finetuned_ORDERED_BY_ID_NFC_TITLES/",
    #     gpu=True,
    #     onlytitles=True)

        self.nermodel = UFALNERExtractor(ner_model)

        self.nextdoc()

    def nextdoc(self, id_=None):
        if id_ is None:
            id_ = self.spp.sampleid()
        self.id = id_
        self.initid = self.id # initially selected

    def getdoc(self, id_=None):
        id_ = self.id if id_ is None else id_
        did = self.db.id2did(id_)
        doc = self.db.get_block_texts(did)
        title = doc.popitem(last=False)
        date_ = self.db[id_]["date"]        
        return id_, did, title, doc, date_

    def save(self, cause, fname="claims.jsonl"):
        logger.info(f"saving '{cause}' to: {fname}")
        with open(fname, "a") as f:
            f.write(f"""{json.dumps({
                'id': self.id, 
                'claim': self.claim, 
                'refs_semantic': self.refs_semantic, 
                'refs_ner': self.refs_ner,
                'cause': cause})}\n""")
