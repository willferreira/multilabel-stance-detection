import os
import sys
import tempfile
import subprocess
from subprocess import PIPE
import time

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from utils import decode_powerset_labels, encode_powerset_labels, encode_binary_relevance_labels,\
    decode_binary_relevance_labels


# FASTTEXT_HOME should be set to dir/folder of fasttext binary, e.g.
#
# C:\\Users\\wferr\\fasttext-win64-latest-Release\\Release\\
# or
# /Users/williamferreira/dev/fastText-0.2.0/
_fasttext_home = os.environ['FASTTEXT_HOME']
print('FASTTEXT_HOME={}'.format(_fasttext_home))

# set up link to fasttext binary
if sys.platform == 'win32':
    _fasttext_exe = '{}\\fasttext.exe'.format(_fasttext_home)
else:
    _fasttext_exe = '{}/fasttext'.format(_fasttext_home)

# fasttext commands
_cmd_fit = [_fasttext_exe, 'supervised']
_cmd_predict_prob = [_fasttext_exe, 'predict-prob']


class FastTextClassifierWrapper(BaseEstimator, ClassifierMixin):
    """A class that wraps the FastText binaries, enabling the FastText model
    to be used by sklearn GridSearchCV."""
    def __init__(self, columns, dim=300, epoch=100, word_ngrams=1):
        """
        Construct a new FastText wrapper
        :param columns: pd.Index, columnsof the original y DataFrame containing the stance labels
        :param dim: int, dimension of the word embeddings
        :param epoch: int, number of epochs to train for
        :param word_ngrams: int, the number of word nrgams to use
        """
        self.columns = columns
        self.dim = dim
        self.epoch = epoch
        self.word_ngrams = word_ngrams

    @staticmethod
    def _make_tmp_files():
        """Generate some intermediate files for FastText"""
        fh, name = tempfile.mkstemp()
        os.close(fh)
        os.remove(name)

        train_filename = name + '_train.txt'
        model_filename = name + '_model'

        fh, name = tempfile.mkstemp()
        os.close(fh)
        os.remove(name)

        predict_filename = name + '_pred_.txt'
        return train_filename, model_filename, predict_filename

    @staticmethod
    def _remove_tmp_files(train_filename, model_filename, predict_filename):
        """Clean up intermediate files required by FastText as they can get a bit big"""
        try:
            os.remove(train_filename)
            os.remove(model_filename + '.bin')
            os.remove(model_filename + '.vec')
            os.remove(predict_filename)
        except:
            pass

    def _fit(self, train_filename, model_filename):
        command = _cmd_fit + ['-input', train_filename,
                              '-output', model_filename,
                              '-epoch', str(self.epoch),
                              '-wordNgrams', str(self.word_ngrams),
                              '-dim', str(self.dim)]
        subprocess.run(command, check=True)

    @staticmethod
    def _predict(model_filename, predict_filename):
        command = _cmd_predict_prob + [model_filename + '.bin', predict_filename]
        return subprocess.run(command, check=True, stdout=PIPE)


class FastTextClassifierPowerset(FastTextClassifierWrapper):

    def __init__(self, columns, dim=300, epoch=100, word_ngrams=1):
        super(FastTextClassifierPowerset, self).__init__(columns, dim, epoch, word_ngrams)
        self.train_filename = None
        self.model_filename = None
        self.predict_filename = None

    def fit(self, X, y, sample_weight=None):
        self.train_filename, self.model_filename, self.predict_filename = self._make_tmp_files()

        encoded_labels = encode_powerset_labels(y)

        with open(self.train_filename, 'w', encoding='utf-8') as train_data_file:
            for i in range(X.shape[0]):
                train_data_file.write(encoded_labels[i] + ' ' + X[i, 0] + '\n')
            train_data_file.flush()

        self._fit(self.train_filename, self.model_filename)

    def predict(self, X):
        with open(self.predict_filename, mode='w', encoding='utf-8') as predict_data_file:
            for utterance in X[:, 0]:
                predict_data_file.write(utterance + '\n')
            predict_data_file.flush()

        # predict labels and convert back into multi-label binary values
        y_pred = self._predict(self.model_filename, self.predict_filename)

        yp = [tuple(s.split(' ')) for s in y_pred.stdout.decode("utf-8").split('\n')]
        yp = list(filter(lambda x: len(x) == 2, yp))
        y_pred = decode_powerset_labels([l[0] for l in yp])

        self._remove_tmp_files(self.train_filename, self.model_filename, self.predict_filename)

        return pd.DataFrame(data=y_pred, columns=self.columns)


class FastTextClassifierBinaryRelevance(FastTextClassifierWrapper):

    def __init__(self, columns, dim=300, epoch=100, word_ngrams=1):
        super(FastTextClassifierBinaryRelevance, self).__init__(columns, dim, epoch, word_ngrams)
        self.train_filename = {}
        self.model_filename = {}
        self.predict_filename = {}

    def _create_files(self):
        for i in range(len(self.columns)):
            tf, mf, pf = self._make_tmp_files()
            self.train_filename[i] = tf
            self.model_filename[i] = mf
            self.predict_filename[i] = pf

    def fit(self, X, y, sample_weight=None):
        self._create_files()

        y_encoded = encode_binary_relevance_labels(y)
        for j in range(y_encoded.shape[1]):
            with open(self.train_filename[j], 'w', encoding='utf-8') as train_data_file:
                for i in range(y_encoded.shape[0]):
                    train_data_file.write(y_encoded[i, j] + ' ' + X[i, 0] + '\n')
                train_data_file.flush()

                self._fit(self.train_filename[j], self.model_filename[j])
            # leave the disk alone for a few seconds
            time.sleep(2)

    def predict(self, X):
        results = {}

        for j in range(len(self.columns)):
            with open(self.predict_filename[j], mode='w', encoding='utf-8') as predict_data_file:
                for utterance in X[:, 0]:
                    predict_data_file.write(utterance + '\n')
                predict_data_file.flush()

                # predict labels and convert back into multi-label binary values
                y_pred = self._predict(self.model_filename[j], self.predict_filename[j])
                results[j] = decode_binary_relevance_labels(y_pred)[:, 0]

        # clean up
        for i in range(len(self.columns)):
            self._remove_tmp_files(self.train_filename[i], self.model_filename[i],
                                   self.predict_filename[i])

        results = pd.DataFrame(data=results)
        results.columns = self.columns
        return results