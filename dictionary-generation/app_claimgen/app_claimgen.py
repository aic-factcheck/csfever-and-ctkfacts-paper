import streamlit as st

import argparse
from collections import OrderedDict
import datetime as dt
import humanize
import json
import numpy as np
import os
from os.path import join as pjoin
import pandas as pd
import re
import torch

from utils.tokenization import detokenize, detokenize2

from ner import UFALNERExtractor, find_ners
from state import State
from dictionary_semantic import search_by_clustering
from dictionary_ner import search_ner

import logging
logger = logging.getLogger(__name__)

@st.cache(allow_output_mutation=True)   
def initialize(ner_model, db_name, kw_model, sem_model, sem_embeddings, sem_faiss_index, excludekw):
    return State(ner_model, db_name, kw_model, sem_model, sem_embeddings, sem_faiss_index, excludekw=excludekw)

def formatdate(date_):
    return date_.strftime('%-d.%-m. %Y (%H:%M:%S)')

def annotate_words(txt, awords):
    # awords = [w.lower() for w in awords]
    awords.sort(key=lambda s: len(s), reverse=True) # dirty trick to deal with subword overlaps

    def split_helper(txt, aidx):
        if len(txt) == 0:
            return []
        if aidx >= len(awords):
            return [txt]
        aword = awords[aidx]
        # st.write(f"aword: '**{aword}**'")
        # st.write(f"txt: {txt}")
        
        segments = re.split(aword, txt, flags=re.IGNORECASE) # split by aword
        # st.write(segments)

        # put annotated awords between all segments, an call recursively for next aword
        res  = split_helper(segments[0], aidx+1)
        for i in range(1, len(segments)):
            # res.append(annotation(aword, "NER"))
            res.append(f"**{aword}**")
            res += split_helper(segments[i], aidx+1)
        return res
        
    segments = split_helper(txt, 0)
    return "".join(segments)

def main(state):
    stylehtml = """
        <style>
            div[role="radiogroup"] .st-bp {
                padding-bottom: 8px;
            }
            div[role="radiogroup"] label:nth-child(2n) {
                background-color: #f0f2f6;
            }
        </style> 
        """
    st.markdown(stylehtml, unsafe_allow_html=True)

    st.sidebar.title('Claim Generation')
    claim = st.sidebar.text_area("Claim:")

    if st.sidebar.button("Save"):
        st.info("claim saved")
        state.claim = claim
        state.save("ok")
        load_next = True
        state.nextdoc()
    
    if st.sidebar.button("Next"):
        state.claim = ""
        state.save("fail")
        state.nextdoc() 

    with st.sidebar.beta_expander("Semantic Dictionary", True):
        sem_k = st.slider("Clusters (k)", min_value=1, max_value=10, value=3, step=1)
        sem_prek = st.slider("Preretrieve for Clustering (prek)", min_value=1, max_value=1024, value=1024, step=1)
        sem_niter = st.slider("K-Means iterations", min_value=10, max_value=500, value=10, step=1)
        sem_npts = st.slider("Documents per Cluster", min_value=1, max_value=10, value=3, step=1)
        sem_randompts = st.checkbox("Random Cluster Points", False, key="Random Cluster Points Semantic")
        sem_olderonly = st.checkbox("Older Only", False, key="Older Only Semantic")
        sem_sort = st.checkbox("Sort", True, key="Sort Semantic")
        sem_notitles = st.checkbox("No Titles", True, key="No Titles Semantic")

    with st.sidebar.beta_expander("NER Dictionary", True):
        ner_olderonly = st.checkbox("Older Only", False, key="Older Only NER")
        ner_notitles = st.checkbox("No Titles", True, key="No Titles NER")
    
    force_id = st.sidebar.text_input("Force ID", state.initid)
    if st.sidebar.button("Force"):
        state.nextdoc(force_id)

    lcol, rcol = st.beta_columns(2)

    srcid, srcdid, (_, srctitle), doc, srcdate = state.getdoc()
    lcol.subheader(srctitle)
    lcol.markdown(f"*{formatdate(srcdate)}, {srcdid}*")

    ids = [id_ for id_ in doc.keys()]
    blocks = [detokenize2(block) for block in doc.values()]

    try:
        initidx = ids.index(state.initid)
    except ValueError:
        st.warning(f"can't find {state.initid}")
        initidx = 0
    idx = lcol.radio("select source paragraph", range(len(blocks)), format_func=lambda e: blocks[e], index=initidx)
    state.id = ids[idx]
    state.refs_semantic = []
    state.refs_ner = []
    
    search_term = srctitle + " " + blocks[idx]
    
    ners = find_ners(state, search_term, exclude_ners=["ČTK"])

    rcol.markdown(f"""**Searching for title and block {idx}**:
- {annotate_words(srctitle, ners)}
- {annotate_words(blocks[idx], ners)}""")
    rcol.markdown("**NERs:** " + ", ".join(ners))
    # for block in doc["blocks"]:
        # lcol.text(block)

    def show_block(id_, date_, srcdate, context):
        tdelta = srcdate - date_
        
        if tdelta.days < 0:
            tdeltastr =  humanize.precisedelta(-tdelta, minimum_unit="days") + " after"
        else:
            tdeltastr =  humanize.precisedelta(tdelta, minimum_unit="days") + " before"
        
        if date_ >= srcdate:
            st.markdown(f'<font color="red">*{formatdate(date_)}*, {tdeltastr}</font>', unsafe_allow_html=True)
        else:
            st.markdown(f'<font color="green">*{formatdate(date_)}*, {tdeltastr}</font>', unsafe_allow_html=True)
        st.write(blocks[id_])
        if st.button(id_, key=f"Button {id_}_{context}"):
            state.claim = ""
            state.save("search")
            state.nextdoc(id_)
            st.experimental_rerun()


    res = search_by_clustering(state, state.id, search_term, k=sem_k, prek=sem_prek, npts=sem_npts, niter=sem_niter, 
        olderonly=sem_olderonly, sort=sem_sort, randompts=sem_randompts, notitles=sem_notitles)
    with rcol.beta_expander(f"Semantic Dictionary ({len(res['semantic_blocks'])})", True):
        # rcol.subheader("Semantic Dictionary")
        for par in res["semantic_blocks"]:
            id_, blocks, title, did, date_ = par["id"], par["blocks"], par["title"], par["did"], par["date"]
            state.refs_semantic.append(id_)
            st.markdown(f'**{title}**')
            show_block(id_, date_, srcdate, "sem") 

    res = search_ner(state, state.id, ners, olderonly=ner_olderonly, notitles=ner_notitles)
    with rcol.beta_expander(f"NER Dictionary ({len(res['ner_blocks'])})", True):
        # rcol.subheader("NER Dictionary")
        for par in res["ner_blocks"]:
            state.refs_ner.append(id_)
            id_, blocks, title, did, date_, search_term = par["id"], par["blocks"], par["title"], par["did"], par["date"], par["search_term"]
            st.markdown(f'Terms: **{search_term}**  \n**{title}**')
            show_block(id_, date_, srcdate, "ner")


parser = argparse.ArgumentParser()
parser.add_argument('--ner_model', required=True, type=str, help='location of NameTag2 NER model')
parser.add_argument('--db_name', required=True, type=str, help='SQLite page database /path/to/fever.db')
parser.add_argument('--kw_model', required=True, type=str, help='keyword (NER) model location, e.g., DRQA index file')
parser.add_argument('--sem_model', required=True, type=str, help='semantic model type, e.g., "bert-base-multilingual-cased" or model dir')
parser.add_argument('--sem_embeddings', required=True, type=str, help='PyTorch tensor embedding file/dirctory, e.g., /path/to/embedded_pages.pt, if not given')
parser.add_argument('--sem_faiss_index', required=True, type=str, help='FAISS index specification for the semantic model')
parser.add_argument('--excludekw', required=False, type=str, default="", help='keywords to exclude separated by semicolon "sport;burza", case insensitive')

try:
    args = parser.parse_args()
except SystemExit as e:
    logger.error(e)
    os._exit(e.code)

st.beta_set_page_config(layout="wide")
state = initialize(args.ner_model, args.db_name, args.kw_model, args.sem_model, args.sem_embeddings, args.sem_faiss_index, excludekw=args.excludekw)
main(state)
