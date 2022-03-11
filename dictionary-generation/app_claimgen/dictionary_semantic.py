from collections import OrderedDict
import faiss
import numpy as np
import scipy

import logging
logger = logging.getLogger(__name__)

from utils.tokenization import detokenize, detokenize2

def search_by_clustering(state, srcid, search_term, k, prek=1024, npts=1, niter=10, olderonly=True, notitles=False, sort=True, randompts=False):
        # retrieve prek vector closest to the "search_term"
        block_ids, block_scores, claim_embedding, block_embeddings = state.semmodel.retrieve(search_term, k=prek, embeddings=True)
        if olderonly:
            mask = (block_ids < srcid) # take older only
            # mask = (block_ids < srcid) & np.array([state.db.istitle(id_) for id_ in block_ids]) # only titles
    #         mask = (block_ids < srcid) & np.array([not db.istitle(srcid) for id_ in block_ids])  # take only older and titles
        else:
            mask = (block_ids != srcid)

        if notitles:
            mask = mask & np.array([not state.db.istitle(id_) for id_ in block_ids])

        block_ids = block_ids[mask]
        block_scores = block_scores[mask]
        block_embeddings = block_embeddings[mask, :]
        logger.debug(f"block_embeddings.shape: {block_embeddings.shape}")
        k = k if k < block_embeddings.shape[0] else block_embeddings.shape[0]  
        logger.debug(f"k: {k}")
        
        # find k clusters in this prek points
        embdim = claim_embedding.shape[0] 
        kmeans = faiss.Kmeans(embdim, k, niter=niter, verbose=False)
        kmeans.train(block_embeddings)
        
        if randompts:
            # select random npts points from each cluster 
            D, I = kmeans.index.search(block_embeddings, 1)
            I = I.ravel()
            allindices = np.arange(len(I))
            idxs = []
            for i in range(k):
                clusteridxs = allindices[I == i]
                if len(clusteridxs) > npts:
                    clusteridxs = np.random.choice(clusteridxs, npts)
                idxs.append(clusteridxs)
            idxs = np.hstack(idxs)
        else:
            # find npts closest documents per each cluster centroid
            index = faiss.IndexFlatL2(embdim)
            index.add(block_embeddings)
            D, I = index.search(kmeans.centroids, npts)
            idxs = I.ravel()
        
        sel_embeddings = block_embeddings[idxs, :]

        if sort:
            # rearange indices according to distance from the claim embedding
            distances = scipy.spatial.distance.cdist(claim_embedding.reshape(1, -1), sel_embeddings, "cosine")[0]
            idxs = idxs[np.argsort(distances)]
        
        ids = [block_ids[i] for i in idxs]
        
        res = OrderedDict()
        res["semantic_blocks"] = []
        srcdid = state.db.id2did(srcid)
        srcdate = state.db.id2date(srcid)
        srctxt = state.db.get_block_text(srcid)
        unique_txts = set(srctxt) # not interested in repeated block texts (unique ids are not enough)
        for id_ in ids:
            if id_ not in state.db.id2row: # might be filtered out
                continue
            did = state.db.id2did(id_)
            date_ = state.db.id2date(id_)
            if srcdid == did: # do not allow another block from the same document
                continue
            if olderonly and date_ >= srcdate:
                continue 
            blocks = OrderedDict([(k, detokenize2(v)) for k, v in state.db.get_block_texts(did).items()])
            sem_txt = blocks[id_]
            if sem_txt not in unique_txts: # blocks are often copied to other document - we don't want them in dictionary
                unique_txts.add(sem_txt) 
                title = list(blocks.values())[0]
                res["semantic_blocks"].append({"id": id_, "did": did, "blocks": blocks, "title": title, "date": date_})
        return res
