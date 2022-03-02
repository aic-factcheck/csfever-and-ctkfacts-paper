import argparse


parser = argparse.ArgumentParser(description='Process arguments')
parser.add_argument('--claims', type=str, default=None, 
                    help='Path to the csv with claims.')
parser.add_argument('--labels', type=str, default=None, 
                    help='Path to the csv with labels.')
parser.add_argument('--data', type=str, default=None, 
                    help='Path to the folder with jsonl files.')
parser.add_argument('--cues', default='unigram', type=str, choices=['unigram', 'bigram', 'wordpiece', 'all'],
                    help='Valid representation of cues: unigram(s), bigram(s), wordpiece(s).')
parser.add_argument('--skip', default=4, type=int,
                    help='Number of skips in generating skipgrams. Relevant only for DCI metric calculation.')
parser.add_argument('--cv', default=False, action='store_true', 
                    help='CrossValidation like sample 10 subdatasets')
parser.add_argument('--export', type=str, 
                    help='Export all the results (not depends on topk argument) into a given folder.')
parser.add_argument('--split', type=str, default='all', choices=['train', 'dev', 'test', 'all'],
                    help="Data split.")
parser.add_argument('--tokenizer', type=str, default='bert-base-multilingual-cased', 
                    help="Tokenizer used for getting wordpieces.")
parser.add_argument('--seed', type=int, default=11)
parser.add_argument('--morphodita', type=str, default=None, 
                    help='MorphoDita path. If the path is given, then only negations are used as cues.')
parser.add_argument('--negation_only', default=False, action='store_true', 
                    help='Return only negations extracted by MorphoDita')
parser.add_argument('--save_latex', default=False, action='store_true', 
                    help='Save results into latex files.')