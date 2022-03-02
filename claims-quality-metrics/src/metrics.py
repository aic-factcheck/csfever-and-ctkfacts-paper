from collections import Counter, OrderedDict
import sys
import pandas as pd

from nltk.util import skipgrams
import math
from typing import Tuple, Optional, List, Any, Dict


### --------------------------------------------------------------------------------------------------------------------
### Applicability, Productivity, Utility Part
### --------------------------------------------------------------------------------------------------------------------
def create_unigrams(data: pd.DataFrame) -> List[List[str]]:
    """Convert claims (sentences) to unigrams"""
    # if morphodita:
    #     return [[ii.strip('.').strip() for ii in claim.split() if ii.strip('.').strip() and morphodita.is_negation(ii.strip('.').strip())] for claim in data['claim'] if isinstance(claim, str)]
    # else:
    return [[ii.strip('.').strip() for ii in claim.split() if ii.strip('.').strip()] for claim in data['claim'] if isinstance(claim, str)]


def create_bigrams(data: pd.DataFrame) -> List[List[str]]:
    """Convert claims (sentences) to bigrams"""
    return [[i.strip('.') + ' ' + ii.strip('.')
            for i, ii in zip(claim.split()[:-1], claim.split()[1:])] 
            for claim in data['claim'] if isinstance(claim, str)]


def create_wordpieces(data: pd.DataFrame, tokenizer: Any) -> List[List[str]]:
    """Convert claims (sentences) to wordpieces"""
    return [tokenizer.tokenize(claim.rstrip('.')) for claim in data['claim'] if isinstance(claim, str)]


def get_cues(data: List[pd.DataFrame], cue: str, tokenizer: Any, logger: Any) -> List[List[List[str]]]:
    """
    Prepares cues (unigrams, bigrams or wordpieces). 
    Cue is a lexical unit representing the potential bias in the data (e.g. 'not' in english data that occurs mainly
    in the claims labeled as 'REFUTES').
    
    Input:
        data:   list of dataframes (each dataframe represents a fold)
        cue:    cue representation

    Return:
        Nested lists of strings - Folds[Claims[Cues[str]]]
    """
    ret = None
    if cue == 'unigram':
        ret = [create_unigrams(sample) for sample in data]
    elif cue == 'bigram':
        ret = [create_bigrams(sample) for sample in data]
    elif cue == 'wordpiece' and tokenizer:
        ret = [create_wordpieces(sample, tokenizer) for sample in data]
    else:
        logger.error("Probably missing tokenizer!")
        sys.exit(0)
    return ret


def get_applicability(cues: List[List[List[str]]]) -> List[Counter]:
    """
    Input:
        cues: nested lists of strings - Folds[Claims[Cues[str]]] - containing cues

    Return:
        List with counted applicability for each cue. 
        Either a single Counter or k Counters in case of cross-validation setup.

    Note:
        Cue Applicability = the absolute number of claims in the dataset that contain the cue irrespective of their label 
    """
    def calculate_applicability(cues: List[List[str]]):
        tmp = []  # type: ignore
        for ii in [set(i) for i in cues]:
            tmp += ii
        return Counter(tmp)

    applicability = [calculate_applicability(cues_in_fold) for cues_in_fold in cues]
    return applicability


def get_productivity(data: List[pd.DataFrame], cues: List[List[List[str]]], 
                     applicability: List[Counter]) -> List[Dict[str, int]]:
    """
    Input:
        data: list of dataframes (each dataframe represents a fold)
        cues: nested lists of strings - Folds[Claims[Cues[str]]] - containing cues

    Return:
        List with counted productivity for each cue. 
        Either a single OrderedDict or k OrderedDicts in case of cross-validation setup.
        
        e.g. OrderedDict: 
            {   
                'car': {'SUPPORT': 4, 'REFUTES': 3, 'NOT ENOUGH INFO': 7},
                'dog': {'SUPPORT': 7, 'REFUTES': 8, 'NOT ENOUGH INFO': 12}
            }

    Note:
        Cue Productivity = is the frequency of the most common label across the claims that contain the cue 
    """
    def get_max(values: Dict[str, int]) -> Tuple[Optional[str], int]:
        label, max_count = None, 0
        for k, v in values.items():
            if v > max_count:
                max_count = v
                label = k
        return label, max_count

    def calculate_productivity(df: pd.DataFrame, cues: List[List[str]], applicability: Counter) -> Dict[str, int]: 
        counts_per_cue = {}  # type: ignore
        for i, words in enumerate([set(i) for i in cues]):
            for w in words:
                if w not in counts_per_cue:
                    counts_per_cue[w] = {}
                if df['label'][i] not in counts_per_cue[w]:
                    counts_per_cue[w][df['label'][i]] = 1
                else:
                    counts_per_cue[w][df['label'][i]] += 1

        max_counts = {k: get_max(v) for k, v in counts_per_cue.items()}
        productivity = {k: (v[0], v[1] / applicability[k]) for k, v in max_counts.items()} 
        return OrderedDict(sorted(productivity.items(), key=lambda kv: kv[1], reverse=True))  # type: ignore

    productivity = [calculate_productivity(data[i], cues[i], applicability[i]) for i in range(len(data))]
    return productivity


def get_coverage(data: List[pd.DataFrame], applicability: List[Counter]) -> List[Dict]:
    """
    Input:
        data:           list of dataframes (each dataframe represents fold)
        applicability:  either a single Counter or k Counters in case of cross-validation setup 
                        with counted applicability for each cue

    Return:
        List with counted coverage for each cue. 
        Either a single Dict or k Dicts in case of cross-validation setup.

    Note:
        Cue Coverage    = applicability of a cue / total number of claims
                        = tells in how many claims is the cue present
    """
    def calculate_coverage(df: pd.DataFrame, applicability: Counter) -> Dict:
        return {k: v / len(df) for k, v in applicability.items()}

    coverage = [calculate_coverage(data[i], applicability[i]) for i in range(len(data))]
    # sorted_cov = [OrderedDict(sorted(coverage[i].items(), key=lambda kv: kv[1], reverse=True)) 
    #                 for i in range(len(data))]  # je toto k necemu? 
    return coverage


def get_utility(productivity: List[Dict[str, int]], num_labels: int) -> List[Dict]:
    """
    Input:
        productivity:   list of OrderedDicts with counted productivity for each cue in each of the k folds
        num_labels:     number of unique labels in the dataset

    Return:
        List with counted utility for each cue. 
        Either a single Dict or k Dicts in case of cross-validation setup.

    Note:
        Cue Utility = the higher utility the easier decision for ML alg

        Helps to compare metrics between different datasets. A cue is only useful to 
        a machine learning model if productivity_k > 1 / m, where m is the number of possible labels.
    """
    utility = [{k: v[1] - 1/num_labels for k, v in productivity[i].items()} for i in range(len(productivity))]
    return utility


def get_result_frame(data: List[pd.DataFrame], cue_form: str, num_unique_labels: int,
                     tokenizer: Any, logger) -> pd.DataFrame:
    """Prepare the final dataframe with computed metrics"""
    def create_result_frame(productivity: Dict[str, int], utility: Dict[str, int], coverage: Dict[str, int]) -> pd.DataFrame:
        res = pd.DataFrame.from_dict(productivity, orient='index', columns=['most_freq_label', 'productivity']).join(
                [pd.DataFrame.from_dict(utility, orient='index', columns=['utility']),
                pd.DataFrame.from_dict(coverage, orient='index', columns=['coverage'])])

        res['harmonic_mean'] = res.apply(lambda x: 2 / (1/x['productivity'] + 1/x['coverage']), axis=1)
        return res.sort_values('harmonic_mean', ascending=False)

    # Calculate the metrics
    cues = get_cues(data, cue_form.lower().strip(), tokenizer, logger)
    applicability = get_applicability(cues)
    productivity = get_productivity(data, cues, applicability)
    coverage = get_coverage(data, applicability)
    # print(cues[0][:2], '\n', applicability[0]['v'], '\n', productivity[0]['v'], '\n', coverage[0]['v'])
    utility = get_utility(productivity, num_unique_labels)

    res_folds = [create_result_frame(productivity[i], utility[i], coverage[i]) for i in range(len(data))]
    # Merge all the samples and compute the estimate
    numeric_cols = ['productivity', 'utility', 'coverage', 'harmonic_mean']
    res = res_folds[0]
    for i in range(1, len(data)):
        res[numeric_cols] = res[numeric_cols].add(res_folds[i][numeric_cols], fill_value=0)
    res[numeric_cols] = res[numeric_cols].div(len(data))
    res = res.sort_values('harmonic_mean', ascending=False).round(4)
    res = res.reset_index()
    res = res.rename(columns={'index': 'cue'})
    return res

### --------------------------------------------------------------------------------------------------------------------
### DCI Part
### --------------------------------------------------------------------------------------------------------------------
def claim_to_unigrams(claim: str) -> List[str]:
    """Convert claims to unigrams"""
    return [ii.strip('.') for ii in claim.split()] if isinstance(claim, str) else None

def claim_to_wordpieces(claim: str, tokenizer: Any) -> List[str]:
    """Convert claim to wordpieces"""
    return tokenizer.tokenize(claim.rstrip('.')) if isinstance(claim, str) else None

def create_skipgrams(data: pd.DataFrame, cue: str, skip: int, tokenizer: Any = None) -> Tuple[Dict[str, Dict[str, int]],
                                                                                              Dict[str, int],
                                                                                              Dict[str, int],
                                                                                              int]:
    """
    Creates skipgrams and cue counts per label and per document. Cue is represented by a skipgram.
    Input:
        data:       dataframe with cues
        cue:        cue representation
        skip:       number of skips for skipgram generation
        tokenizer:  tokenizer

    Return:
        skipgrams_per_label = {'V Hradci': {'Supports': 4, 'Refutes': 1}, ...}
        skipgrams_total = {'V Hradci': 5, ...}
        skipgrams_document_frequency = {'V Hradci': 2, ...}
        total_documents = total number of documents

    Note: 
        if skip == 4, then skipgrams function generates all the skipgrams with 0, 1, 2, 3 and 4 skipped tokens.
    """
    skipgrams_per_label, skipgrams_total = {}, {}  # type: ignore
    skipgrams_document_frequency, tmp_skipgrams_df, total_documents = {}, {}, len(data['claim'])  # type: ignore
    rep2int = {'unigram': 1, 'wordpiece': 1, 'bigram': 2, 'trigram': 3}
    for i, claim in enumerate(data['claim']):
        # TODO rewrite -- added expost and slightly dumb? (calculating same thing as in the applicability?!) 
        _skipgrams = (skipgrams(claim.split(), rep2int[cue], skip) if rep2int[cue] > 1 else 
                    claim_to_unigrams(claim) if cue == 'unigram' else claim_to_wordpieces(claim, tokenizer))
        for skipgram in _skipgrams:
            skipgram = " ".join(list(skipgram)) if rep2int[cue] > 1 else "".join(list(skipgram))
            # Count skipgrams per cue
            if skipgram in skipgrams_total:
                skipgrams_total[skipgram] += 1
            else:
                skipgrams_total[skipgram] = 1
            if skipgram in skipgrams_per_label:
                if data['label'][i] in skipgrams_per_label[skipgram]:
                    skipgrams_per_label[skipgram][data['label'][i]] += 1
                else:
                    skipgrams_per_label[skipgram][data['label'][i]] = 1
            else:
                skipgrams_per_label[skipgram] = {data['label'][i]: 1}
            
            # Count document frequency per cue
            if skipgram in tmp_skipgrams_df:
                tmp_skipgrams_df[skipgram].add(i)
            else:
                tmp_skipgrams_df[skipgram] = set([i])
                    
    # Count the distinct docs         
    for k, v in tmp_skipgrams_df.items():
        skipgrams_document_frequency[k] = int(len(v))

    return skipgrams_per_label, skipgrams_total, skipgrams_document_frequency, total_documents


def compute_normalised_dist(nominator: Dict[str, Dict[str, int]], denominator: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    """Returns normalised distribution over cues and labels"""
    return {cue: 
            {label: count / total for label, count in nominator[cue].items()} 
            for cue, total in denominator.items()}


def entropy(x: Dict[str, float]) -> float:
    return sum([v * math.log(v, 10) for k, v in x.items()])


def lambda_h(N: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Computes information based factor (entropy)"""
    h = {k: 1 + entropy(v) for k, v in N.items()}
    return h


def lambda_f(s: int, doc_freq_per_cue: Dict[str, int], total_docs: int) -> Dict[str, float]:
    """
    Computes frequency-based scaling factor equivalent to normalized/scaled document frequency of a cue.
    Simple explanation: the number of documents in which is the cue present.
    """
    f = {k: math.pow((v / total_docs), (1/s)) for k, v in doc_freq_per_cue.items()}
    return f


def DCI(lamh: Dict[str, float], lamf: Dict[str, float]) -> Dict[str, float]:
    """Computes Dataset-weighted Cue Information"""
    dci = {k: math.sqrt(vh * lamf[k]) for k, vh in lamh.items()}
    return dci


def get_dci_result_frame(data: pd.DataFrame, cue: str = 'bigram', skip: int = 4, 
                         hyperpar_s: int = 3, tokenizer: Any = None) -> pd.DataFrame:
    """Computes Dataset-weighted Cue Information"""
    skipgrams_label, skipgrams_total, skipgrams_df, skipgrams_total_docs = create_skipgrams(data, cue, 4, tokenizer)
    N = compute_normalised_dist(skipgrams_label, skipgrams_total)
    lambh = lambda_h(N)
    lambf = lambda_f(hyperpar_s, skipgrams_df, skipgrams_total_docs)
    dci = DCI(lambh, lambf)
    # dci = sorted(dci.items(), key=lambda kv: kv[1], reverse=True)  # dict -> list of tuples
    dci_sorted = sorted(dci.items(), key=lambda kv: kv[1], reverse=True)
    dci_df = pd.DataFrame(dci_sorted, columns=['cue', 'DCI'])
    return dci_df.round(4)
