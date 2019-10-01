from abc import ABC
from collections import Counter

import numpy as np
import re

import nltk
from nltk.tokenize.nist import NISTTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, make_union


_punctuation = re.compile('^[^a-zA-Z0-9_]$')


class StatelessTransform(ABC):
    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)


# Definitions of the Simaki paper features - these are also defined in the Dan's MSc thesis in
# features.feature_transformers.py, but here I've split them out into separate Transforms to
# make them more explicit, at the expense of some inefficiency and redundancy.


class AverageWordLength(StatelessTransform):
    """Average word length"""

    def transform(self, X, y=None):
        awl = []
        tokenizer = NISTTokenizer()
        for i in range(X.shape[0]):
            tokens = tokenizer.tokenize(X[i, :][0], lowercase=True)
            awl.append(np.mean([len(w) for w in tokens if not _punctuation.match(w)]))
        return np.array(awl).reshape(-1, 1)


class ConjunctionFrequency(StatelessTransform):
    """Conjunction frequency"""

    def transform(self, X, y=None):
        cf = []
        tokenizer = NISTTokenizer()
        for i in range(X.shape[0]):
            tokens = tokenizer.tokenize(X[i, :][0], lowercase=True)
            cf.append(len(list(filter(lambda x: x[1] == 'CC', nltk.pos_tag(tokens)))))
        return np.array(cf).reshape(-1, 1)


class SentenceLengthInWords(StatelessTransform):
    """Sentence length in words"""

    def transform(self, X, y=None):
        sliw = []
        tokenizer = NISTTokenizer()
        for i in range(X.shape[0]):
            tokens = tokenizer.tokenize(X[i, :][0], lowercase=True)
            sliw.append(len(tokens))
        return np.array(sliw).reshape(-1, 1)


class CommaFrequency(StatelessTransform):
    """Comma frequency"""

    def transform(self, X, y=None):
        cf = []
        tokenizer = NISTTokenizer()
        for i in range(X.shape[0]):
            tokens = tokenizer.tokenize(X[i, :][0], lowercase=True)
            cf.append(len(list(filter(lambda x: x[1] == ',', nltk.pos_tag(tokens)))))
        return np.array(cf).reshape(-1, 1)


class FullStopFrequency(StatelessTransform):
    """Full stop frequency"""

    def transform(self, X, y=None):
        fsf = []
        tokenizer = NISTTokenizer()
        for i in range(X.shape[0]):
            tokens = tokenizer.tokenize(X[i, :][0], lowercase=True)
            fsf.append(len(list(filter(lambda x: x[1] == '.', nltk.pos_tag(tokens)))))
        return np.array(fsf).reshape(-1, 1)


class HapaxLegomena(StatelessTransform):
    """Hapax Legomena (number of words appearing in utterance only once)"""

    def transform(self, X, y=None):
        hl = []
        tokenizer = NISTTokenizer()
        for i in range(X.shape[0]):
            tokens = tokenizer.tokenize(X[i, :][0], lowercase=True)
            c = Counter([w for w in tokens if not _punctuation.match(w)])
            hl.append(len([w for w, c in c.items() if c == 1]))
        return np.array(hl).reshape(-1, 1)


class NumberDistinctWords(StatelessTransform):
    """Number of different words used"""

    def transform(self, X, y=None):
        ndw = []
        tokenizer = NISTTokenizer()
        for i in range(X.shape[0]):
            tokens = tokenizer.tokenize(X[i, :][0], lowercase=True)
            ndw.append(len(set([w for w in tokens if not _punctuation.match(w)])))
        return np.array(ndw).reshape(-1, 1)


class SentenceLengthInCharacters(StatelessTransform):
    """Sentence length in characters"""

    def transform(self, X, y=None):
        slic = []
        for i in range(X.shape[0]):
            sentence = X[i, :][0]
            slic.append(len(set([c for c in sentence if not _punctuation.match(c)])))
        return np.array(slic).reshape(-1, 1)


class PunctuationFrequency(StatelessTransform):
    """Punctuation frequency"""

    def transform(self, X, y=None):
        pf = []
        tokenizer = NISTTokenizer()
        for i in range(X.shape[0]):
            tokens = tokenizer.tokenize(X[i, :][0], lowercase=True)
            pf.append(len(list(filter(lambda x: x[1] in ('.', ','), nltk.pos_tag(tokens)))))
        return np.array(pf).reshape(-1, 1)


class HapaxDisLegomena(StatelessTransform):
    """Hapax dislegomena (number of words appearing in utterance only twice)"""

    def transform(self, X, y=None):
        hl = []
        tokenizer = NISTTokenizer()
        for i in range(X.shape[0]):
            tokens = tokenizer.tokenize(X[i, :][0], lowercase=True)
            c = Counter([w for w in tokens if not _punctuation.match(w)])
            hl.append(len([w for w, c in c.items() if c == 2]))
        return np.array(hl).reshape(-1, 1)


class BoW(StatelessTransform):
    """Unigrams"""

    def __init__(self, n_grams=1):
        self.n_grams = n_grams

    def transform(self, X, y=None):
        tokenizer = NISTTokenizer()
        tokenized = np.array([' '.join(tokenizer.tokenize(X[i, :][0], lowercase=True)) for i in range(X.shape[0])])
        vectorizer = CountVectorizer(ngram_range=(1, self.n_grams), max_features=500, stop_words='english')
        return vectorizer.fit_transform(tokenized).toarray()


_transforms = [
    AverageWordLength(),
    ConjunctionFrequency(),
    SentenceLengthInWords(),
    CommaFrequency(),
    FullStopFrequency(),
    HapaxLegomena(),
    NumberDistinctWords(),
    SentenceLengthInCharacters(),
    PunctuationFrequency(),
    HapaxDisLegomena(),
    BoW(2)
]


def make_baseline_pipeline(clf):
    union = make_union(*[t for t in _transforms])
    return make_pipeline(*[
        union,
        StandardScaler(),
        clf
    ])