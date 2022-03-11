from collections import OrderedDict
import json
import os
from typing import Union, Tuple, List, Iterable, Dict, Optional
from transformers import BertTokenizer

class ICTBertTokenizer(BertTokenizer):
    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )
        
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        assert token_ids_1 is None
        res = super().build_inputs_with_special_tokens(token_ids_0, token_ids_1)[:-1] # cut the last [SEP] symbol
        return res

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        assert token_ids_1 is None
        res = super().create_token_type_ids_from_sequences(token_ids_0, token_ids_1)[:-1] # cut the last [SEP] symbol
        return res
