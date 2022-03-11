
from collections import OrderedDict
import json
from pathlib import Path

import datasets
from datasets import ClassLabel

import hashlib

from utils.tokenization import MorphoDiTaTokenizer

import logging
logger = logging.getLogger(__name__)

# TODO: modified code from summarization/sumeczech.py

class IdnesCommentsImporter:
    def __init__(self, datadir="data/idnes", tokenizer=None, version=2, k=8, method=""):
        self.datadir = datadir
        self.tokenizer = tokenizer
        self.sentence_tokenizer = MorphoDiTaTokenizer(lang="cs")
        self.version = version
        self.k = k
        self.method = method

    def preprocess_inputs(self, texts, encoder_max_length, prefix="", split_input_sentences=False):
        # some models need prefixes
        if prefix != "":
            texts = [prefix + t for t in texts]

        # for mBART: puts </s> after end of each sentence
        if split_input_sentences:
            texts = [self.tokenizer.eos_token.join(
                self.sentence_tokenizer.tokenizeSentences(t)) for t in texts]

        inputs = self.tokenizer(texts, padding="max_length",
                                truncation=True, max_length=encoder_max_length)
        return inputs

    def _preprocess_batch(self, batch, text_column, summary_column, encoder_max_length, decoder_max_length,
                          prefix, split_input_sentences, split_output_sentences):
        # tokenize the inputs and labels
        if text_column == "ftext":
            texts = [abstract + " " + text for abstract,
                     text in zip(batch["abstract"], batch["text"])]
            batch[text_column] = texts
        elif text_column == "htext":
            texts = [headline + ". " if headline.endswith(".") else " " + text for headline,
                     text in zip(batch["headline"], batch["text"])]
            batch[text_column] = texts
        elif text_column == "habstract":
            texts = [headline + ". " if headline.endswith(".") else " " + abstract for headline,
                     abstract in zip(batch["headline"], batch["abstract"])]
            batch[text_column] = texts
        else:
            texts = batch[text_column]

        inputs = self.preprocess_inputs(texts, encoder_max_length, prefix=prefix, split_input_sentences=split_input_sentences)

        with self.tokenizer.as_target_tokenizer():
            if split_output_sentences:
                summaries = [self.tokenizer.eos_token.join(
                    self.sentence_tokenizer.tokenizeSentences(t)) for t in batch[summary_column]]
            else:
                summaries = batch[summary_column]
            outputs = self.tokenizer(summaries, padding="max_length",
                                     truncation=True, max_length=decoder_max_length)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [[-100 if token == self.tokenizer.pad_token_id else token for token in labels]
                           for labels in batch["labels"]]

        return batch

    def raw(self):
        v = self.version
        k = self.k
        m = self.method
        trn_file = str(Path(self.datadir, f"idnes_comments-v{v}-k{k}{m}-train.json"))
        dev_file = str(Path(self.datadir, f"idnes_comments-v{v}-k{k}{m}-dev.json"))
        tst_file = str(Path(self.datadir, f"idnes_comments-v{v}-k{k}{m}-test.json"))

        logger.info(f"reading trn file: {trn_file}")
        logger.info(f"reading dev file: {dev_file}")
        logger.info(f"reading tst file: {tst_file}")

        dset = datasets.load_dataset(
            "json", data_files={"trn": trn_file, "dev": dev_file, "tst": tst_file})
        return dset

    def preprocess(self, text_column, summary_column,
                   encoder_max_length, decoder_max_length,
                   max_trn=None, max_dev=None, max_tst=None,
                   remove_columns=True,
                   prefix="",
                   split_input_sentences=False, split_output_sentences=False, load_from_cache_file=True):
        assert self.tokenizer is not None, "Missing a tokenizer needed for the preprocessing! Set it in IdnesCommentsImporter constructor!"
        dset = self.raw()

        if max_trn is not None:
            dset["trn"] = dset["trn"].select(range(max_trn))
        if max_dev is not None:
            dset["dev"] = dset["dev"].select(range(max_dev))
        if max_tst is not None:
            dset["tst"] = dset["tst"].select(range(max_tst))

        keep_columns = ["input_ids", "attention_mask",
                        "decoder_input_ids", "decoder_attention_mask", "labels"]
        if remove_columns:
            rm_columns = list(
                dset["trn"].features.keys() - set(keep_columns))
        else:
            rm_columns = []
        logger.info(f"columns to remove: {remove_columns}")

        logger.info("preprocessing data")
        fn_kwargs = OrderedDict([
            ("text_column", text_column),
            ("summary_column", summary_column),
            ("encoder_max_length", encoder_max_length),
            ("decoder_max_length", decoder_max_length),
            ("prefix", prefix),
            ("split_input_sentences", split_input_sentences),
            ("split_output_sentences", split_output_sentences),
        ])

        for split in ["trn", "dev", "tst"]:

            fingerprint_args = OrderedDict([
                ("tokenizer.name_or_path", self.tokenizer.name_or_path),
                ("max_trn", max_trn),
                ("max_dev", max_dev),
                ("max_tst", max_tst),
                ("split", split)
            ])
            fingerprint_args.update(fn_kwargs)
            fingerprint = hashlib.md5(
                bytes(json.dumps(fingerprint_args), "utf-8")).hexdigest()

            dset[split] = dset[split].map(
                self._preprocess_batch,
                batched=True,
                batch_size=1024,
                # num_proc=4,
                remove_columns=rm_columns,
                load_from_cache_file=load_from_cache_file,
                desc=f"tokenization of {split}",
                fn_kwargs=fn_kwargs,
                new_fingerprint=fingerprint
            )

        dset.set_format(
            type="torch", columns=keep_columns, output_all_columns=not rm_columns
        )

        return dset
