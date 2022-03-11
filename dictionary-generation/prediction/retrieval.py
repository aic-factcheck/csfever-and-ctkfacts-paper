from datetime import date
from drqa import retriever
import faiss
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
import numpy as np
import os
import pathlib
from pathlib import Path
import sqlite3
import time
import unicodedata as ud

from embed.embeddings import load_embeddings
from models.transformers import load_model
from utils.sem_distance import sem_distance
from utils.tokenization import detokenize, detokenize2

from pyserini.search import SimpleSearcher

from colbert.evaluation.loaders import load_colbert
from colbert.modeling.inference import ModelInference
from colbert.ranking.rankers import Ranker
from colbert.indexing.faiss import get_faiss_index_name

import os
import random
import queue
import threading
from collections import OrderedDict, defaultdict
import itertools

import logging
logger = logging.getLogger(__name__)

class AbstractRetriever:
    pass

class AbstractFilteringRetriever(AbstractRetriever):
    pass

class FilteringRetriever(AbstractFilteringRetriever):
    def __init__(self, db, model, maxk=1000, max_repeats=3, initial_scale=5):
        # takes any DR `model` and tries to deliver exact number of documents in line with filtering options as required in `retrieve()`
        self.db = db
        self.model = model
        self.maxk = maxk
        self.max_repeats = max_repeats
        self.initial_scale = initial_scale

    def retrieve(self, claim, k, max_titles=0, datemin=date.min, datemax=date.max, **kwargs):
        # does similar thing to search.py#filter_results but it is better to move it here
        # TODO: add support for datemin/datemax
        allres = []
        print(f"k={k} for: {claim}")
        for i in range(self.max_repeats):
            k2 = min(self.initial_scale * k * (i + 1), self.maxk)
            print(f"k2={k2}")
            results = self.model.retrieve(claim, k2, **kwargs)
            resblocksset = set()
            ntitles = 0
            for result in results:
                id_ = result["id"]
                if (ntitles < max_titles or not self.db.istitle(id_)) and (datemin <= self.db.id2date(id_).date() <= datemax):
                        blocktxt = self.db.get_block_text(id_)
                        if blocktxt not in resblocksset:
                            allres.append(result)
                            resblocksset.add(blocktxt) # no duplicates (there are often multiple copies of a single paragraph)
                            if self.db.istitle(id_):
                                ntitles += 1
            if len(allres) >= k or k2 == self.maxk:
                break
        return allres[0:k]


class TopNDocsDRQA(AbstractRetriever):
    def __init__(self, db, model):
        self.db = db
        self.model = model
        self.ranker = retriever.get_class('tfidf')(tfidf_path=self.model)
        self.name = f"DRQA"
        
    def retrieve(self, claim, k, preprocess=lambda t: t):
        st = time.time()
        doc_names, doc_scores = self.ranker.closest_docs(preprocess(ud.normalize("NFD", claim)), k)
        self.duration = time.time() - st
        # logger.info(f"doc_names: {doc_names}")
        # logger.info(f"doc_scores: {doc_scores}")
        return [{"id": id_, "score": {"orig": score}, "search": "drqa"} for id_, score in zip(doc_names, doc_scores)]
    
class TopNDocsDRQATwoTower(AbstractRetriever):
    def __init__(self, db, premodel, model_dir):
        self.db = db
        self.premodel = premodel
        self.model = load_model(model_dir)
        self.model.eval()
        self.name = f"{self.premodel.name}+{os.path.basename(model_dir)}"
        
    def retrieve(self, claim, k, prek=500, preprocess=lambda t: t):
        st = time.time()
        doc_names_pre, doc_scores_pre = self.premodel.retrieve(claim, k=prek)
        logger.info(f"pre-selected {len(doc_names_pre)} documents")
        txts = []
        for did in doc_names_pre:
            txt = self.db.get_doc_text(did)
            if txt is None:
                logger.warning(f"document {did} has no text!")
            else:
                txts.append(txt)
        logger.info(f"imported text for {len(txts)} pre-selected documents")

        txts = [claim] + txts
        txts = map(preprocess, txts)
        x = [detokenize2(txt) for txt in txts]
        self.preduration = time.time() - st
        logger.info("TT input ready")

        st2 = time.time()
        y = self.model.encode(x, convert_to_numpy=True)
        logger.info("TT model evaluated")

        y_claim = np.tile(y[0:1], (y.shape[0]-1, 1))
        y_pages = y[1:]
        dists = paired_cosine_distances(y_claim, y_pages)
        inds = np.argsort(dists)[:k]
        doc_names, doc_scores = [doc_names_pre[i] for i in inds], [dists[i] for i in inds]
        self.modelduration = time.time() - st2
        self.duration = time.time() - st

        # logger.info(f"doc_names: {doc_names}")
        # logger.info(f"doc_scores: {doc_scores}")

        return [{"id": id_, "score": {"orig": score}, "search": "drqatt"} for id_, score in zip(doc_names, doc_scores)]

class TopNDocsTwoTowerFaiss(AbstractRetriever):
    def __init__(self, db_name, model_dir, embeddings, db_table="documents", norm='NFC', gpu=True, faissindex='Flat', onlytitles=False):
        self.model = load_model(model_dir)
        self.model.eval()
        embeddings = Path(embeddings)
        self.embeddings = load_embeddings(embeddings)
        n, embedding_dim = self.embeddings.shape
        logger.info("{} embeddings of dimension {}".format(n, embedding_dim))
        with sqlite3.connect(db_name) as conn:
            # index must be bypassed so page ids are read in a same ordering as text when page embedding were computed!
            sql = f"SELECT id FROM {db_table} WHERE bid = 0" if onlytitles else f"SELECT id FROM {db_table}"
            self.corpus_pages = sorted(map(lambda e: e[0], conn.execute(sql)))
        logger.info(f"imported {len(self.corpus_pages)} ids from {db_name}")
        assert n == len(self.corpus_pages)
        self.norm = norm

        logger.info("indexing using Faiss")
        index = faiss.index_factory(embedding_dim, faissindex, faiss.METRIC_INNER_PRODUCT)
        logger.info(" ... normalizing")
        faiss.normalize_L2(self.embeddings)
        if gpu:
            logger.info(" ... configuring GPU")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, index) # use gpu
        else:
            logger.info(" ... configuring CPU")
            self.index = index
        logger.info('training index...')
        self.index.train(self.embeddings)
        logger.info('adding embeddings')
        self.index.add(self.embeddings)

        faiss_file = Path(embeddings.parent, f"{embeddings.name}.faiss")
        # logger.info(f"saving to: {faiss_file}")
        # index = faiss.index_gpu_to_cpu(self.index)
        # faiss.write_index(index, faiss_file)
        logger.info(" ... done")

        self.name = os.path.basename(model_dir)
        
    def retrieve(self, claim, k, preprocess=lambda t: t, embeddings=False):
        st = time.time()
        claim = preprocess(ud.normalize(self.norm, claim))
        claim_embedding = self.model.encode([claim], show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(claim_embedding)
        distances, idxs = self.index.search(claim_embedding, k)
        doc_names = [self.corpus_pages[i] for i in idxs[0]]
        doc_scores = distances[0].tolist()
        self.duration = time.time() - st
        # logger.info(f"doc_names: {doc_names}")
        # logger.info(f"doc_scores: {doc_scores}")
        if embeddings:
            return np.array(doc_names), np.array(doc_scores), claim_embedding[0], self.embeddings[idxs[0], :]
        return [{"id": id_, "score": {"orig": score}, "search": "drqattfaiss"} for id_, score in zip(doc_names, doc_scores)]
    

class Anserini(AbstractRetriever):
    def __init__(self, model_dir="/mnt/data/factcheck/CTK/par5/index/anserini"):
        self.index = model_dir
        self.k1 = 0.6
        self.b = 0.5

        self.searcher = SimpleSearcher(str(self.index))
        
    def retrieve(self, query, k):
        assert isinstance(query, str), "Expected string input as a query!"
        print(f'Initializing BM25, setting k1={self.k1} and b={self.b}', flush=True)

        hits = self.searcher.search(query, k)
        ret_ids, ret_scores = [], []
        for i in range(len(hits)):
            ret_ids.append(hits[i].docid)
            ret_scores.append(hits[i].score)
        return [{"id": id_, "score": {"orig": score}, "search": "anserini"} for id_, score in zip(ret_ids, ret_scores)]


class ColBERTArgs:
    def __init__(self):
        self.query_maxlen = 32
        self.similarity = 'cosine'
        self.rank = -1
        self.faiss_name = None
        self.faiss_depth = 512
        self.part_range = None
        self.depth = 100  # number of returned top-k documents

        self.amp = True
        self.doc_maxlen = 180
        self.mask_punctuation = True
        self.bsize = 32
        self.dim = 64
        self.nprobe = 32
        self.partitions = 32768
        
        self.index_root = "/mnt/data/factcheck/CTK/par5/colbert/indexes"
        # self.index_name = "ctk-fever-v2.1.L2.32x200k"
        self.index_name = "ctk-fever-64dim.L2.32x200k"
        # self.checkpoint = "/mnt/data/factcheck/CTK/par5/colbert/ctk-fever-v2.1/train.py/ctk-fever-v2.1.l2/checkpoints/colbert.dnn"
        self.checkpoint = "/mnt/data/factcheck/CTK/par5/colbert/ctk-fever-64dim/train.py/ctk-fever-64dim.l2/checkpoints/colbert.dnn"
        self.root = "/mnt/data/factcheck/CTK/par5/colbert"
        # self.experiment = "ctk-fever-v2.1"
        self.experiment = "ctk-fever-64dim"
        self.idConvPath = "/mnt/data/factcheck/CTK/par5/interim/old-id2new-id.tsv"
        self.qrels=None

class ColBERT:
    def __init__(self, args: ColBERTArgs = None):
        self.args = args if args else ColBERTArgs()
        self.args.colbert, self.args.checkpoint = load_colbert(self.args)
        self.args.index_path = os.path.join(self.args.index_root, self.args.index_name)
        if self.args.faiss_name is not None:
            self.args.faiss_index_path = os.path.join(self.args.index_path, self.args.faiss_name)
        else:
            self.args.faiss_index_path = os.path.join(self.args.index_path, get_faiss_index_name(self.args))

        self.inference = ModelInference(self.args.colbert, amp=self.args.amp)
        self.ranker = Ranker(self.args, self.inference, faiss_depth=self.args.faiss_depth)
        
        with open(self.args.idConvPath) as fr:
            self.colbId2parId = {int(l.split('\t')[1].strip()): l.split('\t')[0] for l in fr if l.strip()}
        
    def idx2par(self, ids):
        """Converts ColBERT inner indices to paragraph indices"""
        ret = [self.colbId2parId[i] if int(i) in self.colbId2parId else None for i in ids]
        return ret
        
    def retrieve(self, query, k=None, convert_ids=True):
        assert isinstance(query, str), "Expected string input as a query!"
        k = k if k else self.args.depth
        Q = self.ranker.encode([query])
        pids, scores = self.ranker.rank(Q)
        
        ret_ids, ret_scores = [], []
        for pid, score in itertools.islice(zip(pids, scores), k):
            ret_ids.append(pid)
            ret_scores.append(score)
        if convert_ids:
            ret_ids = self.idx2par(ret_ids)
        # return ret_ids, ret_scores
       
        return [{"id": id_, "score": {"orig": score}, "search": "colbert"} for id_, score in zip(ret_ids, ret_scores)]

class MetaRetriever(AbstractFilteringRetriever):
    def __init__(self, db, semmodel: AbstractFilteringRetriever, kwmodel:AbstractFilteringRetriever, nlimodels, scoring_model=None, max_titles=0, sort_nli=True):
        self.db = db
        self.semmodel = semmodel
        self.kwmodel = kwmodel
        self.nlimodels = nlimodels
        self.max_titles = max_titles
        self.scoring_model = scoring_model # should be semantic model
        self.sort_nli = sort_nli

    def retrieve(self, query, k, **kwargs):
        semres = self.semmodel.retrieve(query, k, **kwargs)
        kwres = self.kwmodel.retrieve(query, k, **kwargs)
        
        # now merge them
        allres = semres.copy()
        allidsset = set([r["id"] for r in allres])
        for kwitem in kwres:
            if kwitem["id"] not in allidsset:
                allres.append(kwitem)
                allidsset.add(kwitem["id"])
        allids = [r["id"] for r in allres]
        print(f"retrieved SEM: {len(semres)}, KW: {len(kwres)}, MERGED: {len(allids)}")

        # compute common score for both semantic and keyword search
        if self.scoring_model is not None:
            common_scores = sem_distance(self.scoring_model, query, allids, preprocess=detokenize2)
            for res, common in zip(allres, common_scores):
                res["score"]["orig"] = common

        nli_sentences = [[detokenize2(self.db.get_block_text(block)), detokenize2(query)] for block in allids]
        nlires = [nlimodel.predict(nli_sentences, apply_softmax=True) for nlimodel in self.nlimodels]
        nlires = np.mean(nlires, axis=0) # NLI ensemble predictions
        confs = np.max(nlires[:, 0:2], axis=1) # confidences for REFUTES, SUPPORTS
        idxs = np.argsort(confs)[::-1] # sort from the highest
        idxs = idxs[:k]

        allres = [allres[i] for i in idxs]
        nlires = nlires[idxs, :]

        if self.sort_nli:
            classes = np.argmax(nlires, axis=1)
            # refutes, supports, nei -> supports, refutes, nei
            c2c = {0:1, 1:0, 2:2}
            classes = [c2c[c] for c in classes]
            idxs = np.argsort(classes)
            allres = [allres[i] for i in idxs]
            nlires = nlires[idxs, :]


        for res, nli in zip(allres, nlires):
            res["nli"] = {"refutes": nli[0], "supports": nli[1], "nei": nli[2]}

        return allres
