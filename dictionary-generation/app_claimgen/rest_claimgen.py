import argparse
from collections import OrderedDict
from flask import Flask, jsonify, request
import torch
import werkzeug

from dictionary_semantic import search_by_clustering
from dictionary_ner import search_ner
from ner import find_ners
from state import State

import logging
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/sample', methods=['GET'])
def sample():
    id_, did, (title_id, title), blocks, date_ = app.state.getdoc()
    app.state.nextdoc()
    return jsonify(id=id_, did=did, blocks=blocks, date=str(date_), title=title)

@app.route('/fetch/<id>', methods=['GET'])
def fetch_doc(id):
    id_, did, (title_id, title), blocks, date_ = app.state.getdoc(id)
    return jsonify(id=id_, did=did, blocks=blocks, date=str(date_), title=title)

@app.route('/dictionary/<id>', methods=['GET'])
def dictionary(id):
    # id: source block id
    sem_prek = request.args.get('prek', default=1024, type=int) # this number of block is preseleted by BERT
    ner_limit = request.args.get('nerlimit', default=3, type=int) # max number of ner search results
    sem_k = request.args.get('k', default=3, type=int) # number of k-means clusters built on "prek" preselected blocks
    sem_niter = request.args.get('niter', default=10, type=int) # the number of k-means iterations
    sem_npts = request.args.get('npts', default=3, type=int) # return this number of blocks per cluster 
    sem_randompts = request.args.get('randompts', default=0, type=int) # 0/1: 0-select random "npts" points from each cluster, 1-find "npts" closest documents per each cluster centroid  
    notitles = request.args.get('notitles', default=1, type=int) # 0/1: filter out blocks which are titles (id = "*_0")
    olderonly = request.args.get('older', default=1, type=int) # keep only blocks of documents older than the source block (given by "id")
    tosort = request.args.get('sort', default=1, type=int) # 0/1: sort the semantic dictionary part based on embedding similarity to the source block (given by "id")  

    id_, did, (title_id, title), doc, date_ = app.state.getdoc(id)
    search_term = request.args.get('q', default=title + " " + doc[id_], type=str)
    semantic_results = search_by_clustering(app.state, id_, search_term, k=sem_k, prek=sem_prek, 
        npts=sem_npts, niter=sem_niter, olderonly=bool(olderonly), sort=bool(tosort), 
        randompts=bool(sem_randompts), notitles=bool(notitles))
        
    ners = find_ners(app.state, search_term, exclude_ners=["ČTK"])
    ner_results = search_ner(app.state, id_, ners, olderonly=bool(olderonly), notitles=bool(notitles), limit=ner_limit)

    return jsonify(
        semantic_blocks=semantic_results['semantic_blocks'],
        ners=ners,
        ner_blocks=ner_results['ner_blocks'])


if __name__ == '__main__':
    # query examples:
    #   curl -i http://localhost:8601/sample
    #   curl -i http://localhost:8601/dictionary/T201311150323901_1
    #   curl -i http://localhost:8601/dictionary/T201311150323901_1?k=5&npts=5&older=1&sort=1
    log_fmt = '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument('--ner_model', required=True, type=str, help='location of NameTag2 NER model')
    parser.add_argument('--db_name', required=True, type=str, help='SQLite page database /path/to/fever.db')
    parser.add_argument('--kw_model', required=True, type=str, help='keyword (NER) model location, e.g., DRQA index file')
    parser.add_argument('--sem_model', required=True, type=str, help='semantic model type, e.g., "bert-base-multilingual-cased" or model dir')
    parser.add_argument('--sem_embeddings', required=True, type=str, help='PyTorch tensor embedding file/dirctory, e.g., /path/to/embedded_pages.pt, if not given')
    parser.add_argument('--sem_faiss_index', required=True, type=str, help='FAISS index specification for the semantic model')
    parser.add_argument('--excludekw', required=False, type=str, default="", help='keywords to exclude separated by semicolon "sport;burza", case insensitive')
    args = parser.parse_args()

    if app.debug:
        logger.info("running in DEBUG mode")

    if not app.debug or app.debug and werkzeug.serving.is_running_from_reloader():
        app.state = State(args.ner_model, args.db_name, args.kw_model, 
            args.sem_model, args.sem_embeddings, args.sem_faiss_index, excludekw=args.excludekw)

    app.run(host='0.0.0.0', port=8601)
