from collections import defaultdict, OrderedDict, Counter
import csv
import datetime as dt
from itertools import chain, product
import json
import math
import numpy as np
import os
from os.path import join as pjoin
import pandas as pd
import string
import sklearn
from time import time
from tqdm.autonotebook import tqdm, trange
import unicodedata as ud
import urllib.request


import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, losses, models
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, LabelAccuracyEvaluator, SequentialEvaluator

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from nltk.tokenize import word_tokenize, sent_tokenize

from utils.tokenization import detokenize, detokenize2
from prediction.retrieval import TopNDocsDRQA, TopNDocsDRQATwoTower, TopNDocsTwoTowerFaiss
from models.milos import MILOSPreModel, MILOSModel
from models.transformers import load_model


import logging
logger = logging.getLogger(__name__)


def _import_csvs_to_dicts(ddir, sel_labels):
    # builds several dicts (and other structures) representing the annotated dataset
    # ddir - directory with CSV exported tables from Fcheck annotation platform
    # sel_label

    claims = pd.read_csv(pjoin(ddir, "claim.csv"))
    # take only mutations not original claims
    claims = claims.dropna(subset=['mutated_from'])
    logger.warning(f"# claims imported: {len(claims)}")
    if "deleted" in claims.columns:
        delete_mask = claims.deleted == 1
        logger.warning(f" soft deleted: {delete_mask.sum()}")
        claims = claims[~delete_mask].copy()
    labels = pd.read_csv(pjoin(ddir, "label.csv"))
    labels_all = pd.read_csv(pjoin(ddir, "label.csv"))
    labels = labels_all[labels_all.label.isin(sel_labels)]

    evidence = pd.read_csv(pjoin(ddir, "evidence.csv"))
    paragraphs = pd.read_csv(pjoin(ddir, "paragraph.csv"))
    claim_knowledge = pd.read_csv(pjoin(ddir, "claim_knowledge.csv"))

    labeldict = {row.id: {"claim": row.claim, "label": row.label}
                 for _, row in labels.iterrows()}

    claimdict = {row.id: row.claim.strip() for _, row in claims.iterrows()}

    claim2labeltxt = {}
    conflicting_claims = set()
    for _, row in labels_all.iterrows():
        if row.claim in conflicting_claims:
            continue
        if row.claim in claim2labeltxt and claim2labeltxt[row.claim] != row.label:
            # print(f"conflicting claim: id={row.claim} txt=\"{claimdict[row.claim]}\"")
            conflicting_claims.add(row.claim)
            del claim2labeltxt[row.claim]
        claim2labeltxt[row.claim] = row.label
    logger.warning(f"# of claims having conflicting labels: {len(conflicting_claims)}")

     # dict: paragraph_DB_id -> paragraph_id (e.g., "T201710220487401_2")
    blockdict = {
        row.id: f"{row.article}_{row['rank']}" for _, row in paragraphs.iterrows()}
    logger.warning(f"blockdict size: {len(blockdict)}")

    evidencedict = defaultdict(list)
    for _, row in evidence.iterrows():
        evidencedict[row.label].append(blockdict[row.paragraph])

    logger.warning(f"claims: {len(claims.id)}, unique: {len(set(claims.id))}")
    for _, row in claims.iterrows():
        if row.paragraph not in blockdict:
            logger.warning(f"Not found: {row.paragraph}")

    claim2orig_par = {row.id: blockdict[row.paragraph] for _, row in claims.iterrows()}

    ckdict = defaultdict(list)
    for _, row in claim_knowledge.iterrows():
        ckdict[row.claim].append(blockdict[row.knowledge])

    return (
        # DataFrame(id, user, claim, label, sandbox, oracle, flag, condition, created_at, updated_at) filtered to sel_labels
        labels,
        labeldict,  # dict: label_id -> {"claim": claim_id, "label": label_txt}
        claimdict,  # dict: claim_id -> claim_txt
        conflicting_claims,  # set of claim_ids where annotators disagree
        # dict: claim_id -> label_txt (common label, e.g. "SUPPORTS" for non disagreeing annotators)
        claim2labeltxt,
        # dict: label_id -> [paragraph_id1, paragraph_id2, ...], e.g., ['T200708080333101_4', 'T200803130442001_5']
        evidencedict,
        claim2orig_par,  # dict: claim_id -> paragraph_id of paragraph of claim origin
        # dict: claim_id -> [paragraph_id1, paragraph_id2, ..], e.g., ['T200708080333101_4', 'T200803130442001_5'] - dictionary data
        ckdict
    )


def import_csv_dataset(ddir):
    labels, labeldict, claimdict, conflicting_claims, claim2labeltxt, evidencedict, claim2orig_par, ckdict = _import_csvs_to_dicts(
        ddir, sel_labels=["SUPPORTS", "REFUTES"])

    dataset = {}
    for i, labelid in enumerate(labels.id):
        claimid = labeldict[labelid]["claim"]
        if claimid not in claimdict or claimid in conflicting_claims:
            continue
        claimtxt = claimdict[claimid]
        labeltxt = claim2labeltxt[claimid]
        if claimtxt in dataset:
            evidence = dataset[claimtxt]["evidence"]
        else:
            evidence = []
        evidenceset = evidencedict[labelid]

        if len(evidenceset) == 0:
            logger.warning(f"warning evidence set empty for: {labelid}")
            continue
        evidencerec = []
        for e in evidenceset:
            evidencerec.append([labelid, e])
        evidence.append(evidencerec)

        dataset[claimtxt] = {"claim_id": claimid, "label": labeltxt,
                             "claim": claimtxt, "evidence": evidence, "orig_par_id": claim2orig_par[claimid]}
    return dataset


def import_csv_dataset_nei(ddir, n):
    labels, labeldict, claimdict, conflicting_claims, claim2labeltxt, evidencedict, claim2orig_par, ckdict = _import_csvs_to_dicts(
        ddir, sel_labels=["NOT ENOUGH INFO"])

    # distribute dictionary as "evidence" to NEI claims, one at a time; in rounds, always add to all NEI claims (until total of "n" reached)
    dataset = {}
    cnt = 0
    round_cnt = 0
    idx = 0
    round_ = 0
    labels_list = list(labels.id)
    while cnt < n:
        if idx >= len(labels_list):
            idx = 0
            assert round_cnt > 0, f"not enough data, only {cnt} samples available"
            round_ += 1
            logger.info(f"round_cnt {round_cnt} -> new round {round_}")
            round_cnt = 0

        labelid = labels_list[idx]
        claimid = labeldict[labelid]["claim"]

        if claimid not in claimdict or claimid in conflicting_claims:
            idx += 1
            continue

        claimtxt = claimdict[claimid]
        if claimtxt in dataset:
            evidence = dataset[claimtxt]["evidence"]
        else:
            # evidence = [] # creating new evidence sets
            evidence = [[]]  # appending to single ES

        # there is still more in the dictionary for the claim
        if round_ < len(ckdict[claimid]):
            ev = ckdict[claimid][round_]
            if ev not in [e[1] for e in evidence[0]]:
                evidence[0].append([labelid, ev])
            cnt += 1
            round_cnt += 1

        dataset[claimtxt] = {"claim_id": claimid, "label": "NEI",
                             "claim": claimtxt, "evidence": evidence, "orig_par_id": claim2orig_par[claimid]}
        idx += 1
    # some claims may end up with empty dictionaries - remove them
    dataset = {k: v for k, v in dataset.items() if len(v["evidence"][0]) > 0}
    return dataset

def unique_evidence(dataset):
    # removes evidence duplicities, removes other info than paragraph_id (block_id)
    samples = []
    for row in dataset:
        evidence = row["evidence"]
        blocks = [[e[1] for e in ev] for ev in evidence] # keep just block_id
        ublocks = [] # unique blocks
        for b in blocks:
            if b not in ublocks:
                ublocks.append(b)
        newv = row.copy()
        newv["evidence"] = ublocks
        samples.append(newv)
    return samples

def split_evidence_sets(dataset):
    samples = []
    for row in dataset:
        evidence = row["evidence"]
        for ev in evidence:
            newv = row.copy()
            newv["evidence"] = [ev]
            samples.append(newv)
    return samples

def dataset2fever_jsonl(dataset, outfile):
    with open(outfile, 'w', encoding="utf8") as f:
        for r in dataset:
            evidence = r["evidence"]

            verifiable = "NOT VERIFIABLE" if r["label"] == "NEI" else "VERIFIABLE"
            label = "NOT ENOUGH INFO" if r["label"] == "NEI" else r["label"]
            # annotation_id (currently not used), evidence_id, page (paragraph), sentence (always -1)
            evidence = [[[-1, e[0], e[1], -1] for e in ev] for ev in r["evidence"]]
            rec = OrderedDict([
                ("id", r["claim_id"]),
                ("verifiable", verifiable),
                ("label", label),
                ("claim", r["claim"]),
                ("evidence", evidence),
                ("orig_par_id", r["orig_par_id"]), # paragraph of claim origin, not in original FEVER, but important for debugging
            ])
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")

def create_splits(ds, rng):
    # Split based on claim source paragraphs to prevent bleeding between train, test and validation sets.
    pars = sorted(list(set([d["orig_par_id"] for d in ds]))) # sorted so it is reproducible
    pars_trn, pars_tst = sklearn.model_selection.train_test_split(
        pars, train_size=0.8, random_state=rng)
    pars_trn, pars_val = sklearn.model_selection.train_test_split(
        pars_trn, train_size=0.8, random_state=rng)
    pars_trn = set(pars_trn)
    pars_tst = set(pars_tst)
    pars_val = set(pars_val)
    print(
        f"original pararagraphs #, TRN: {len(pars_trn)}, TST: {len(pars_tst)}, VAL: {len(pars_val)}")
    dataset1_trn = [d for d in ds if d["orig_par_id"] in pars_trn]
    dataset1_tst = [d for d in ds if d["orig_par_id"] in pars_tst]
    dataset1_val = [d for d in ds if d["orig_par_id"] in pars_val]
    return dataset1_trn, dataset1_tst, dataset1_val


def convert_to_sentence_pairs(db, dataset, reverse_pairs=False):
    claim_txts = []
    evidence_txts = []
    labels = []
    evidence_idxs = []
    examples = []
    for row in dataset:
        claim = row["claim"]
        label = 0 if row["label"] == 'REFUTES' else 1 if row["label"] == 'SUPPORTS' else 2
        for eset in row["evidence"]:
            #             assert len(eset) == 1 # for now only single document
            id_ = eset[0]
            evidence = " ".join(
                [ud.normalize("NFC", detokenize2(db.get_block_text(id_))) for id_ in eset])
            texts = [ud.normalize("NFC", claim), evidence]
            if reverse_pairs:
                texts.reverse()
            examples.append(InputExample(texts=texts, label=label))
    return examples
