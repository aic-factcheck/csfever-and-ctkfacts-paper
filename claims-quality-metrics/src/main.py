import logging
import datetime
import os
import sys

from args import parser
from utils import process_csv, process_jsonl_in_folder, get_k_fold
from metrics import get_dci_result_frame, get_result_frame
from morphodita import MorphoDiTa

if __name__ == '__main__':
    # Parse arguments
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict['cues'] = args_dict['cues'].lower().rstrip('s').strip()

    # Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Started at {start}")

    given_args = f"\nGIVEN ARGUMENTS\n"
    for k, v in args_dict.items():
        given_args += f"\t{k}: {v}\n"
    logger.info(given_args)

    # Tokenizer
    tokenizer = None
    if args.cues == 'wordpiece':
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            logger.info(f"{tokenizer} succesfully loaded!")
        except ImportError:
            logger.error("Cannot import transformers library -- choose unigram/bigram cue representation instead.")
            sys.exit(0)


    # Check arguments validity
    if args.claims and not os.path.exists(args.claims):
        logging.error(f"Given path to the claims file - {args.claims} - does not exist")
        sys.exit(0)
    elif args.labels and not os.path.exists(args.labels):
        logging.error(f"Given path to the labels file - {args.labels} - does not exist")
        sys.exit(0)
    elif args.data and not os.path.exists(args.data):
        logging.error(f"Given path to the data foler - {args.data} - does not exist")
        sys.exit(0)
    if args.cues.lower().strip() not in ['unigram', 'unigrams', 'bigram', 'bigrams', 'wordpiece', 'wordpieces']:
        logging.error(f"Given cue representation - {args.cues} - is not valid.",
                      f"Valid representation is one of the: ",
                      f"['unigram', 'unigrams', 'bigram', 'bigrams', 'wordpiece', 'wordpieces']")
        parser.print_help()
        sys.exit(0)

    # Load MorphoDita
    mdt = MorphoDiTa(args.morphodita) if args.morphodita else None

    # TODO enable to work with sql dump file as well
    logging.info(f'Processing input files as a single dataset...')
    data = process_jsonl_in_folder(args.data, args.split, logger) if args.data else process_csv(args.claims, args.labels)
    NUM_LABELS = len(data.label.unique())  # num of possible labels (3 for FEVER dataset - supports, refutes, nei)
    logger.info(f"\nNumber of unique labels: {NUM_LABELS}\nNumber of claims: {len(data)}")


    ### ----------------------------------------------------------------------------------------------------------------
    ### Productivity, coverage, utility metrics
    ### ----------------------------------------------------------------------------------------------------------------
    # Prepare data into folds
    if args.cv:
        logging.info(f'Processing csv input files as a dataset of k samples...')
        data_k_folds = get_k_fold(data, k=10, seed=args.seed)  # k = number of folds
    else:
        data_k_folds = [data]

    # Calculate the metrics
    logging.info(f'Computing metrics into the result dataframe...')
    res = get_result_frame(data_k_folds, args.cues, NUM_LABELS, tokenizer, logger)

    # Return only negations extracted by MorphoDita
    if mdt:
        res['negation'] = res['cue'].map(mdt.is_negation)
        if args.negation_only:
            res = res.loc[res["negation"] == True]

    res['productivity'] = res['productivity'].round(2)
    res['coverage'] = res['coverage'].round(2)
    res['harmonic_mean'] = res['harmonic_mean'].round(2)
    res.drop(['utility'], axis=1, inplace=True)
    if 'most_freq_label' in res.columns:
        res['most_freq_label'] = res['most_freq_label'].apply(lambda x: x.replace('SUPPORTS', 'SUP').replace('REFUTES', 'REF').replace('NOT ENOUGH INFO', 'NEI'))

    # Export data
    print(f"\nTOP 20 CUES WITH A HIGHEST POTENTIAL TO CREATE A PATTERN\n{res[:20]}")
    if args.export:
        negation_only = f"-negation_only" if args.negation_only and args.morphodita else ""
        output_path = os.path.join(args.export, f"{args.cues}{negation_only}-util-cov.csv")
        logging.info(f'Saving into file {output_path}')
        if not os.path.exists(args.export):
            try:
                os.mkdir(args.export)
            except Exception as e:
                print("CALCULATION STOPPED!")
                print((f"Given export path {args.export} doesn't exist and cannot be created " +
                       f"(probably multiple non-existing dirs in the path)!"))
                print(e)
                raise e

        res.to_csv(output_path, encoding="utf-8", index=True)

        if args.save_latex:
            output_path = f"{args.cues}{negation_only}-util-cov-latex"
            with open(os.path.join(args.export, output_path), 'w') as fw:
                fw.write(res[:20].to_latex(index=False))

    ### ----------------------------------------------------------------------------------------------------------------
    ### Dataset-weighted Cue Information (DCI)
    ### ----------------------------------------------------------------------------------------------------------------
    # Calculate the metrics
    logger.info("Computing DCI metric")
    dci = get_dci_result_frame(data, args.cues, args.skip, 3, tokenizer)

    # Return only negations extracted by MorphoDita
    if mdt:
        dci['negation'] = dci['cue'].map(mdt.is_negation)
        if args.negation_only:
            dci = dci.loc[dci["negation"] == True]

    # Export data
    print(f"\nTOP 20 CUES PROVIDING HIGHEST INFORMATION GAIN\n{dci[:20]}")
    if args.export:
        s = f"{args.skip}-skip-" if args.cues in ['bigram', "trigram"] else ""
        negation_only = f"-negation_only" if args.negation_only and args.morphodita else ""
        output_path = f"{s}{args.cues}{negation_only}-dci.csv"
        path = os.path.join(args.export, output_path)
        logging.info(f'Saving into file {path}')
        dci.to_csv(path, encoding="utf-8", index=True)

        if args.save_latex:
            output_path = f"{s}{args.cues}{negation_only}-dci-latex"
            with open(os.path.join(args.export, output_path), 'w') as fw:
                fw.write(dci[:20].to_latex(index=False))




