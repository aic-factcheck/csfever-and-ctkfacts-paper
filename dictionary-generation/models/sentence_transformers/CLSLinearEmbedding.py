from collections import OrderedDict
import json
import os
import torch
from torch import Tensor, nn
from typing import Union, Tuple, List, Iterable, Dict, Optional

class CLSLinearEmbedding(nn.Module):
    # implemented according to https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/*.py
    # https://www.sbert.net/docs/training/overview.html#creating-networks-from-scratch
    def __init__(self, word_embedding_dimension: int, target_dimension: int):
        super(CLSLinearEmbedding, self).__init__()
        self.config_keys = ['word_embedding_dimension', 'target_dimension']
        self.word_embedding_dimension = word_embedding_dimension
        self.target_dimension = target_dimension
        self.linear = torch.nn.Linear(word_embedding_dimension, target_dimension)
    
    def forward(self, features: Dict[str, Tensor]):
        x  = features['cls_token_embeddings']        
        output_vector = self.linear(x)
        features.update({'sentence_embedding': output_vector})
        return features
    
    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError()
        
    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}
    
    def save(self, output_path):
        with open(os.path.join(output_path, 'cls_linear_embedding_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)
        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'cls_linear_embedding_config.json'), 'r') as fIn:
            config = json.load(fIn)
        if torch.cuda.is_available():
            weights = torch.load(os.path.join(input_path, 'pytorch_model.bin'))
        else:
            weights = torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        model = CLSLinearEmbedding(**config)
        model.load_state_dict(weights)
        return model