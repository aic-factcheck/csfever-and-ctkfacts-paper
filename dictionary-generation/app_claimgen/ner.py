from ufal.nametag import Ner, Forms, TokenRanges, NamedEntities

from utils.tokenization import detokenize, detokenize2

class UFALNERExtractor:

    def __init__(self, model):
        self.ner = Ner.load(model)
        self.forms = Forms()
        self.tokens = TokenRanges()
        self.entities = NamedEntities()
        self.tokenizer = self.ner.newTokenizer()
        
    def extract(self, claim):
        self.tokenizer.setText(claim)
        ners = []
        nertypes = []
        while self.tokenizer.nextSentence(self.forms, self.tokens):
            self.ner.recognize(self.forms, self.entities)
            
            entities = sorted(self.entities, key=lambda entity: (entity.start, -entity.length))
            
            prev_end = -1
            for entity in entities:
                if (entity.start + entity.length) <= prev_end: # take only the highest level entities
                    continue
                ners.append(" ".join(self.forms[entity.start:entity.start+entity.length]))
                nertypes.append(entity.type)
                prev_end = entity.start + entity.length

        return ners, nertypes

def find_ners(state, search_term, exclude_ners=["ČTK"]):
    ners = state.nermodel.extract(search_term)
    ners = [detokenize2(ner) for ner in ners[0] if ner not in exclude_ners] # detokenize to FIX? detected date
    ners = list(set(ners))
    return ners