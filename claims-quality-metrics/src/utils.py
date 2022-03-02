import pandas as pd
import json
import os
from typing import List


def process_csv(claims: str, labels: str) -> pd.DataFrame:
    """
    Input:
        claims: path to csv file with claims
        labels: path to csv file with labels

    Return:
        pandas dataframe containing merged claims with corresponding labels
    """
    # Read csv files
    _claims = pd.read_csv(claims)
    _claims.dropna(subset=['id', 'claim'], inplace=True)
    _labels = pd.read_csv(labels)

    # Basic data processing
    _labels.id = _labels.claim
    df = _claims.merge(_labels[['id', 'label']], on='id', how='left')
    df = df.dropna(subset=['claim', 'label'])
    df = df[['id', 'claim', 'label']]
    df.reset_index(inplace=True, drop=True)
    return df


def read_jsonl_files(files: List[str]) -> dict:
    """
    Input:
        files: list containing paths of jsonl files

    Return:
        dictionary containing merged claims with corresponding labels
    """
    claims, labels = [], []
    for file in files:
        with open(file) as fr:
            for line in fr:
                d = json.loads(line)
                claims.append(d['claim'])
                labels.append(d['label'])
    assert len(claims) == len(labels)
    # return claims, labels
    ret = {'claim': claims, 'label': labels}
    return ret


def process_jsonl_in_folder(path: str, split: str, logger) -> pd.DataFrame:
    """Read all jsonl files in given path and process them into a single DataFrame."""
    if split == 'all':
        files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".jsonl") and not file.endswith("test.jsonl")]
    else:
        files = [os.path.join(path, split + '.jsonl')]
    logger.info(f"Reading from: {files}.")
    d = read_jsonl_files(files)
    df = pd.DataFrame(d)
    return df


def get_k_fold(data: pd.DataFrame, k: int = 10, seed: int = 11) -> List[pd.DataFrame]:
    """
    Generates k (cross-validation like) random folds.

    Input:
        df: original parsed data
        k:  number of cross-validation sets
    
    Return:
        list of random k subsets.

    Note: 
        This approach assumes a balanced dataset with regard to the frequency of each label. If executed on
        an imbalanced dataset, a given cue’s productivity would be dominated by the most frequent label, 
        not because it is actually more likely to appear in a claim with that label but solely because of the label
        is more frequent in overall. 
    """
    SAMPLE_SIZE = min(data['label'].value_counts())
    SAMPLES = []
    for _ in range(k):
        df_to_join = []
        for label in data.label.unique():
            df_to_join.append(data[data.label == label].sample(SAMPLE_SIZE, random_state=seed)[['claim', 'label']])
        
        SAMPLES.append(pd.concat(df_to_join).reset_index(drop=True))
    return SAMPLES

