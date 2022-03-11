from collections import OrderedDict
import re
import numpy as np

import logging

logger = logging.getLogger(__name__)

from utils.tokenization import detokenize, detokenize2


def search_ner(state, srcid, ners, notitles=False, olderonly=True, limit=10):
    res = OrderedDict()
    res["ners"] = ners

    def extract_pairs_all(ners):
        ner_pairs = []
        for i in range(len(ners) - 1):
            for j in range(i + 1, len(ners)):
                ner_pairs.append(ners[i] + ", " + ners[j])
        return ner_pairs

    search_terms = extract_pairs_all(ners)
    # use TFIDF to find closest matches to the nerpairs
    all_ner_ids = []
    for nerpair in search_terms:
        ids, scores = state.kwmodel.retrieve(nerpair, k=1)
        all_ner_ids.append((ids[0], scores[0]))

    # unique_ids = set(srcid) # not interested in repeated ids
    srctxt = state.db.get_block_text(srcid)
    unique_txts = set(srctxt)  # not interested in repeated block texts (unique ids are not enough)
    srcdid = state.db.id2did(srcid)
    srcdate = state.db.id2date(srcid)

    ner_records = []
    ner_candidates = [(ner_id, score, search_term) for (ner_id, score), search_term in zip(all_ner_ids, search_terms)]
    ner_candidates.sort(key=lambda x: x[1], reverse=True)

    for ner_id, score, search_term in ner_candidates:
        if ner_id not in state.db.id2row:  # may be filtered out while loading DB
            continue
        date_ = state.db.id2date(ner_id)
        if olderonly and date_ >= srcdate:  # take only older TODO fix for exact date comparison
            continue
        if notitles and state.db.istitle(ner_id):
            continue
        if state.db.id2did(ner_id) == srcdid:  # not from the same document as source
            continue
        ner_txt = state.db.get_block_text(ner_id)
        # if ner_id not in unique_ids:
        if ner_txt not in unique_txts:
            # unique_ids.add(ner_id)
            unique_txts.add(ner_txt)
            ner_records.append([ner_id, score, search_term])

    res["ner_blocks"] = []
    for id_, score, search_term in ner_records:
        if limit <= 0:
            break
        limit -= 1
        did = state.db.id2did(id_)
        blocks = OrderedDict([(k, detokenize2(v)) for k, v in state.db.get_block_texts(did).items()])
        title = list(blocks.values())[0]
        res["ner_blocks"].append({
            "id": id_,
            "did": did,
            "score": score,
            "blocks": blocks,
            "title": title,
            "date": state.db.id2date(id_),
            "search_term": search_term,
        })
    return res
