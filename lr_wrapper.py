import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from utils import decode_powerset_labels, encode_powerset_labels


class LogisticRegressionBaselinePowerset(BaseEstimator, ClassifierMixin):
    def __init__(self, columns, C=1.0, penalty='l1', max_iter=2000, random_state=0):
        self.columns = columns
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
        self.random_state=random_state
        self.clf = None

    def fit(self, X, y, sample_weight=None):
        self.clf = LogisticRegression(solver='saga', class_weight='balanced', penalty=self.penalty,
                                      max_iter=self.max_iter, C=self.C, multi_class='multinomial',
                                      random_state=self.random_state)
        self.clf.fit(X, encode_powerset_labels(y))

    def predict(self, X):
        return pd.DataFrame(columns=self.columns,
                            data=decode_powerset_labels(self.clf.predict(X)))
