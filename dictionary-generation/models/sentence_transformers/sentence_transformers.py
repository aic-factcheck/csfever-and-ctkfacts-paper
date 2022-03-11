from collections import OrderedDict
import json
import os
import torch
from torch import Tensor, nn
from typing import Union, Tuple, List, Iterable, Dict, Optional
from sentence_transformers import SentenceTransformer
from models.ICTBertTokenizer import ICTBertTokenizer
from .ICTBert import ICTBert
from .CLSLinearEmbedding import CLSLinearEmbedding

def import_as_sentence_transformer(bert_model, model_weights, max_seq_length=288, embedding_dim=512, strict=True):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = torch.device('cpu')
    # Note this works with transformers==3.0.2 and sentence-transformers=0.3.5.1
    word_embedding_model = ICTBert(bert_model, max_seq_length=max_seq_length)
    pooling_model = CLSLinearEmbedding(word_embedding_model.get_word_embedding_dimension(), embedding_dim)

    if device is not None:
        load_dict = torch.load(model_weights, map_location=lambda storage, loc: storage.cuda(device))
    else:
        load_dict = torch.load(model_weights, torch.device('cpu'))

    # split parameters for BERT and for the last linear pooling layer
    bert_dict = OrderedDict()
    pooling_dict = OrderedDict()
    for k, v in load_dict['model'].items():
        if k.startswith("encoder."):
            bert_dict[k[8:]] = v
        if k.startswith("linear."):
            pooling_dict[k] = v    

    word_embedding_model.bert.load_state_dict(bert_dict, strict=strict)
    pooling_model.load_state_dict(pooling_dict, strict=strict)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model.tokenizer = ICTBertTokenizer.from_pretrained(bert_model)
    return model
