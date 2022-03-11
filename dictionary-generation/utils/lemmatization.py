from ufal.morphodita import Tagger_load, Forms, TaggedLemmas, TokenRanges
from ufal.udpipe import Model, ProcessingError, Sentence
import stanza

"""
Various facades for Czech lemmatization. 
"""

class MorphoDiTaLemmatizer:
    def __init__(self, tagger_file: str):
        self.tagger = Tagger_load(tagger_file)
        self.forms = Forms()
        self.lemmas = TaggedLemmas()
        self.tokens = TokenRanges()
        self.tokenizer = self.tagger.newTokenizer()
        self.morpho = self.tagger.getMorpho()

    def lemmatize(self, text: str):
        self.tokenizer.setText(text)
        while self.tokenizer.nextSentence(self.forms, self.tokens):
            self.tagger.tag(self.forms, self.lemmas)
            for lemma in self.lemmas:
                yield self.morpho.rawLemma(lemma.lemma)

class UDPipeError(Exception):
    """An error which occurred in the :mod:`ufal.udpipe`."""

class UDPipeLemmatizer:
    def __init__(self, udpipe_model: str):
        self.model = Model.load(udpipe_model)
        self.tokenizer = self.model.newTokenizer(self.model.DEFAULT)

    def lemmatize(self, text: str):
        self.tokenizer.setText(text)
        error = ProcessingError()
        sent = Sentence()
        while self.tokenizer.nextSentence(sent, error):
            self.model.tag(sent, self.model.DEFAULT)
            for word in sent.words[1:]:
                yield word.lemma

        if error.occurred():
            raise UDPipeError(error.message)


class StanzaLemmatizer:
    def __init__(self, ):
        self.nlp = stanza.Pipeline(lang='cs', processors='tokenize,mwt,lemma')

    def lemmatize(self, text: str):
        doc = self.nlp(text)
        for sent in doc.sentences:
            for word in sent.words:
                yield word.lemma
