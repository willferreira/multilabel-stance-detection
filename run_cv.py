import os
os.environ['PYTHONHASHSEED'] = str(0)

import tensorflow as tf
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import abc
import click
import pickle
import py
from functools import partial
import itertools as it

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from baseline_features import make_baseline_pipeline
from lr_wrapper import LogisticRegressionBaselinePowerset
from fasttext_wrapper import FastTextClassifierBinaryRelevance, FastTextClassifierPowerset
from mlp_multilabel_wrapper import MultiOutputKerasWrapper, MultilabelMultioutputKerasWrapper,\
    PowersetKerasWrapper, MultitaskKerasWrapper
from utils import get_dataset, get_dataset_cv, jaccard_score_fn, tweet_score_fn, reset_seeds, load_embeddings
from mlp_utils import CrossLabelDependencyLoss, CorrelationLoss


class _ModelRunner(abc.ABC):
    def __init__(self, clf_clz, model_name, dataset_name, verbose=False, n_jobs=1):
        self.clf_clz = clf_clz
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.verbose = verbose
        self.n_jobs = n_jobs

        self.model = None
        self.y_pred = None
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_dataset()

    def get_dataset(self):
        return get_dataset(self.dataset_name)

    def get_dataset_cv(self):
        return get_dataset_cv(self.dataset_name)

    def get_scoring_fn(self):
        return jaccard_score_fn if self.dataset_name == 'bbc' or self.dataset_name.startswith('moral')\
            else tweet_score_fn

    def write_cv_results(self, dir='.'):
        results = {
            'cv_results': self.model.cv_results_,
            'best_params': self.model.best_params_,
            'best_score': self.model.best_score_,
            'y_test': self.y_test,
            'y_pred': self.y_pred,
            'model_name': self.model_name,
            'dataset_name': self.dataset_name
        }

        pth = py.path.local(dir)
        py.path.local(pth).ensure(dir=True)

        filename = 'cv_results_{}_{}.pkl'.format(self.model_name, self.dataset_name)
        with open(pth.join(filename), 'wb') as fp:
            pickle.dump(results, fp)


class _LRPowersetRunner(_ModelRunner):

    def __call__(self, param_grid):
        clf = LogisticRegressionBaselinePowerset(self.y_train.columns, max_iter=1000)
        pipeline = make_baseline_pipeline(clf)

        param_grid = dict(
            logisticregressionbaselinepowerset__C=[0.001, 0.01, 0.1, 1.0, 10],
            logisticregressionbaselinepowerset__penalty=['l1', 'l2'],
            logisticregressionbaselinepowerset__random_state=[0]
        )

        self.model = GridSearchCV(pipeline, cv=self.get_dataset_cv(), param_grid=param_grid,
                                  scoring=self.get_scoring_fn(), verbose=self.verbose, n_jobs=self.n_jobs)
        self.model.fit(self.X_train.values, self.y_train.values)
        self.y_pred = self.model.predict(self.X_test.values)
        return self


class _LRBinaryRelevanceRunner(_ModelRunner):

    def __call__(self, param_grid):
        clf = OneVsRestClassifier(
            LogisticRegression(solver='saga', class_weight='balanced', max_iter=1000, random_state=0)
        )
        pipeline = make_baseline_pipeline(clf)

        self.model = GridSearchCV(pipeline, cv=self.get_dataset_cv(), param_grid=param_grid,
                                  scoring=self.get_scoring_fn(), verbose=self.verbose, n_jobs=self.n_jobs)
        self.model.fit(self.X_train.values, self.y_train.values)

        y_pred = self.model.predict(self.X_test.values)
        self.y_pred = pd.DataFrame(data=y_pred, columns=self.y_test.columns, index=self.X_test.index)
        return self


class _FastTextRunner(_ModelRunner):
    def __init__(self, clf_clz, model_name, dataset_name, verbose=False, n_jobs=1):
        super(_FastTextRunner, self).__init__(clf_clz, model_name, dataset_name, verbose, n_jobs)
        self.clf = clf_clz(columns=self.y_train.columns)

    def __call__(self, param_grid):
        self.model = GridSearchCV(self.clf, cv=self.get_dataset_cv(), scoring=self.get_scoring_fn(),
                                  param_grid=param_grid, verbose=self.verbose, n_jobs=self.n_jobs,
                                  return_train_score=False)
        self.model.fit(self.X_train.values, self.y_train.values)
        self.y_pred = self.model.predict(self.X_test.values)
        return self


class _MLPRunner(_ModelRunner):
    def __init__(self, clf_clz, model_name, dataset_name, verbose=False, n_jobs=1):
        super(_MLPRunner, self).__init__(clf_clz, model_name, dataset_name, verbose, n_jobs)
        self.clf = clf_clz(columns=self.y_train.columns, epochs=50, batch_size=32, verbose=self.verbose)

    def get_dataset(self):
        # Return the dataset appropriate pre-computed embeddings X
        _, _, y_train, y_test = get_dataset(self.dataset_name)
        X_train, X_test = load_embeddings(self.dataset_name)
        return X_train, X_test, y_train, y_test

    def __call__(self, param_grid):
        # all the MLP models get a bit of dropout
        if param_grid is None:
            param_grid = {}

        reset_seeds()
        sess = tf.Session(graph=tf.get_default_graph())
        K.set_session(sess)

        self.model = GridSearchCV(estimator=self.clf, cv=self.get_dataset_cv(), param_grid=param_grid,
                                  scoring=self.get_scoring_fn(), verbose=self.verbose, n_jobs=self.n_jobs)

        self.model.fit(self.X_train.values, self.y_train.values)
        y_pred = self.model.predict(self.X_test.values)
        self.y_pred = pd.DataFrame(data=y_pred)
        self.y_pred.columns = self.y_test.columns

        K.clear_session()
        return self


# all models and their runners
models = {
    'mlp-base': (_MLPRunner, MultiOutputKerasWrapper, None),
    'mlp-powerset': (_MLPRunner, PowersetKerasWrapper, None),
    'mlp-correlation': (_MLPRunner, MultiOutputKerasWrapper,
                        {'loss': [CorrelationLoss(alpha, threshold) for (alpha, threshold) in
                                  it.product(np.arange(0.1, 0.6, 0.1), [0.1, 0.2, 0.3, 0.4, 0.5])]}),
    'mlp-correlation-global': (_MLPRunner, MultiOutputKerasWrapper,
                        {'loss': [CorrelationLoss(alpha=alpha, threshold=threshold, use_y_true=True)
                                  for (alpha, threshold) in
                                  it.product(np.arange(0.1, 0.6, 0.1), [0.1, 0.2, 0.3, 0.4, 0.5])]}),
    'mlp-cross-label-dependency': (_MLPRunner, MultiOutputKerasWrapper,
                                   {'loss': [CrossLabelDependencyLoss(alpha) for alpha in np.arange(0.0, 1.1, 0.1)]}),
    'mlp-multitask': (_MLPRunner, MultitaskKerasWrapper, None),
    'mlp-multilabeloutput': (_MLPRunner, MultilabelMultioutputKerasWrapper,
                             {
                                 'alpha_left': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                                 'alpha_right': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
                              }),
    'mlp-multilabeloutput-base': (_MLPRunner, MultilabelMultioutputKerasWrapper, None),
    'lr-binary-relevance': (_LRBinaryRelevanceRunner, None, {
        'onevsrestclassifier__estimator__C': [0.001, 0.01, 0.1, 1.0, 10],
        'onevsrestclassifier__estimator__penalty': ['l1', 'l2'],
        'onevsrestclassifier__estimator__random_state': [0]
    }),
    'lr-powerset': (_LRPowersetRunner, None, {
            'logisticregressionbaselinepowerset__C': [0.001, 0.01, 0.1, 1.0, 10],
            'logisticregressionbaselinepowerset__penalty': ['l1', 'l2'],
            'random_state': [0]
    }),
    'fasttext-binary-relevance': (_FastTextRunner, FastTextClassifierBinaryRelevance,
                                  {'word_ngrams': [1, 2], 'dim': [300, 512, 1024]}),
    'fasttext-powerset': (_FastTextRunner, FastTextClassifierPowerset,
                          {'word_ngrams': [1, 2], 'dim': [300, 512, 1024]}),
}


# invalid model/dataset combinations
_invalid_model_dataset_combos = {
    ('mlp-multilabeloutput', 'bbc'),
    ('mlp-multitask', 'tweets-DT_HC'),
    ('mlp-multitask', 'tweets-DT_TC'),
    ('mlp-multitask', 'tweets-HC_BS'),
    ('lr-powerset', 'tweets-DT_HC'),
    ('lr-powerset', 'tweets-DT_TC'),
    ('lr-powerset', 'tweets-HC_BS'),
    ('lr-binary-relevance', 'tweets-DT_HC'),
    ('lr-binary-relevance', 'tweets-DT_TC'),
    ('lr-binary-relevance', 'tweets-HC_BS')
}


@click.command()
@click.option('--model-name',
              default='fasttext-powerset',
              type=click.Choice([
                  'mlp-base', 'mlp-powerset', 'mlp-correlation', 'mlp-correlation-global',
                  'mlp-cross-label-dependency', 'mlp-multitask', 'mlp-multilabeloutput',
                  'lr-binary-relevance', 'lr-powerset', 'mlp-multilabeloutput-base',
                  'fasttext-binary-relevance', 'fasttext-powerset', 'all', 'all-lr', 'all-fasttext', 'all-mlp'
              ]),
              help='WARNING: --model-name all runs cv on all models; this can be slow. '
              )
@click.option('--dataset-name',
              default='moral-dataset-MeToo',
              type=click.Choice(['bbc', 'tweets-DT_HC', 'tweets-DT_TC', 'tweets-HC_BS',
                                 'moral-dataset-Baltimore', 'moral-dataset-ALM',
                                 'moral-dataset-Davidson', 'moral-dataset-BLM',
                                 'moral-dataset-Sandy', 'moral-dataset-MeToo',
                                 'moral-dataset-Election']),
              help='--dataset-name tweets runs all tweet data-sets. '
                   'WARNING: the following --model-name/--dataset-name combinations:'
                   '\n{}\n are invalid.'.format(str(_invalid_model_dataset_combos))
              )
@click.option('--verbose/--no-verbose', default=True)
@click.option('--n-jobs', default=1)
def run_cv(model_name, dataset_name, verbose, n_jobs):

    if model_name.startswith('fasttext') and n_jobs > 1:
        n_jobs = 1
        print('WARNING: setting n-jobs=1, since running FastText is not thread-safe')

    if model_name.startswith('all'):
        s = model_name.split('-')
        if len(s) > 1:
            models_to_run = [m for m in models.keys() if m.startswith(s[1])]
        else:
            models_to_run = models.keys()
    else:
        models_to_run = [model_name]
    print('Running models: {} with dataset: {}'.format(models_to_run, dataset_name))
    for m_name in models_to_run:
        if (m_name, dataset_name) in _invalid_model_dataset_combos:
            print('WARNING: Invalid model-name/dataset-name combo: {}, {} - '
                  'IGNORING'.format(model_name, dataset_name))

        runner_clz, clz, param_grid = models[m_name]
        model = runner_clz(clf_clz=clz, model_name=m_name, dataset_name=dataset_name,
                           verbose=verbose, n_jobs=n_jobs)
        model(param_grid).write_cv_results('results')


if __name__ == '__main__':
    run_cv()
