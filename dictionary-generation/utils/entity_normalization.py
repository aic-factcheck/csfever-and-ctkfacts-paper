from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict, Counter

class EntityNormalizer(object):
    """Normalizes named entities by lemmatization and conversion to the root form. 
    The `EntityNormalizer`firstly fits a corpus of entities (see `fit` method) by applying lemmatization and other 
    transforms (currently conversion to the lower case). 
    The `normalize` method then returns an original entity form from the corpus corresponding to the normalized entity form. 
    If the corpus does not contain such original entity, the most common one is selected.
    
    The motivation is to get normalized but still human-acceptable entity identifiers.
    """
    def __init__(self, lemmatizer):
        self._lemmatizer = lemmatizer
        self._original_form = dict()
        self._nentity_counters = defaultdict(Counter) # Counter of original forms for each normalized entity
        self._nentity_hits = Counter() # overall stats: # of original forms per normalized entity 
        
    def _normalize_helper(self, entity: str) -> str:
        return " ".join(self._lemmatizer.lemmatize(entity)).lower()
    
    def fit(self, entities: List[str]) -> None:
        """Fit a corpus to a list of entities."""
        for entity in entities:
            nentity = self._normalize_helper(entity)
            if nentity == entity.lower(): # this seems to be the orginal form matching the normalized norm of the entity
                self._original_form[nentity] = entity
            self._nentity_counters[nentity][entity] += 1
            self._nentity_hits[nentity] += 1
            
    def normalize(self, entity: str):
        """Return normalized version of the entity"""
        nentity = self._normalize_helper(entity)
        if nentity not in self._nentity_counters:
            raise KeyError(f'"{entity}" (normalized as "{nentity}") unknown, did you forget to fit?')
        if nentity in self._original_form:
            return self._original_form[nentity]
        return self._nentity_counters[nentity].most_common(1)[0][0]