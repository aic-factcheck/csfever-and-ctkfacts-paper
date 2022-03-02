from ufal.morphodita import (
    Forms, 
    TokenRanges, 
    TaggedLemmasForms, 
    TaggedLemmas,
    Tagger_load
)


def encode_entities(text: str):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


class MorphoDiTa:
    """
    MorphoDita (https://ufal.mff.cuni.cz/morphodita) class
    """
    def __init__(self, path: str):
        self.tagger = Tagger_load(path)
        self.forms = Forms()
        self.lemmas = TaggedLemmas()
        self.lemmas_forms = TaggedLemmasForms()
        self.tokens = TokenRanges()
        self.tokenizer = self.tagger.newTokenizer()
        self.morpho = self.tagger.getMorpho()

    def lemmatize(self, text):
        self.tokenizer.setText(text)
        while self.tokenizer.nextSentence(self.forms, self.tokens):
            self.tagger.tag(self.forms, self.lemmas)
            for lemma in self.lemmas:
                yield (self.morpho.rawLemma(lemma.lemma), self.morpho.rawLemma(lemma.tag))
                
    def analyze(self, text: str):
        self.tokenizer.setText(text)
        while self.tokenizer.nextSentence(self.forms, self.tokens):
            for form in self.forms:
                result = self.morpho.analyze(form, self.morpho.GUESSER, self.lemmas)
                guesser = "Guesser " if result == self.morpho.GUESSER else ""
                for lemma in self.lemmas:
                    print(f"Form: {form};  Guesser {guesser};  Lemma: {lemma.lemma} Tag: {lemma.tag};  Negation: {lemma.tag[10] == 'N'}")

    def is_negation(self, word: str) -> bool:
        """
            Returns True if a given word is negation.

            Input:
                word: a single word
        """
        assert isinstance(word, str), f"Given word {word} is not a string"
        ret = False
        if len(word.strip().split()) == 1:  # single word
            result = self.morpho.analyze(word, self.morpho.GUESSER, self.lemmas)
            # morphodita can generate more lemmas for a single word (e.g. Nejedlý)
            negative = [lemma.tag[10] == 'N' for lemma in self.lemmas]
            ret = True if any(negative) else False
        return ret