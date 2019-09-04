from functools import partial
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, jaccard_similarity_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, \
    zero_one_loss, jaccard_similarity_score, matthews_corrcoef


# Jaccard score wrapped as a scoring function
jaccard_score_fn = make_scorer(jaccard_similarity_score)


def _tweet_score(y_true, y_pred):
    # the tweet score, as described in
    # https://www.aclweb.org/anthology/E17-2088
    scores = []
    for i in range(y_true.shape[1]):
        scores.append(f1_score(y_true[:, i], y_pred.values[:, i],
                               labels=['FAVOR', 'AGAINST'], average='macro'))
    return np.mean(scores)


tweet_score_fn = make_scorer(_tweet_score)


def get_scoring_function(name, dataset_name):
    assert name in ['jaccard', 'accuracy', 'f1_macro']
    if dataset_name == 'bbc' or dataset_name.startswith('moral'):
        if name == 'jaccard':
            return jaccard_score_fn
        return name
    # need to do something different for tweets as
    # they're multi-class multi-output
    return tweet_score_fn


def reset_seeds(seed_value=0):

    from numpy.random import seed
    seed(seed_value)

    from tensorflow import set_random_seed
    set_random_seed(seed_value)

    import random as rn
    rn.seed(seed_value)


def get_dataset(name):
    if name == 'bbc':
        return get_bbc_dataset()

    if name.startswith('tweets-'):
        dataset = pd.read_csv('{}.csv'.format(name))
        dataset_train = dataset[dataset.set == 'train']
        dataset_test = dataset[dataset.set == 'test']

        X_train = dataset_train[['Tweet']]
        y_train = dataset_train[['Target 1', 'Target 2']]

        X_test = dataset_test[['Tweet']]
        y_test = dataset_test[['Target 1', 'Target 2']]
        return X_train, X_test, y_train, y_test

    dataset = pd.read_csv('{}.csv'.format(name), index_col=0)
    dataset_train = dataset[dataset.set == 'train']
    dataset_test = dataset[dataset.set == 'test']
    X_train = dataset_train[['Tweet']]
    y_train = dataset_train.drop(labels=['Tweet', 'set'] +
                                        [col for col in dataset.columns if col.startswith('fold')], axis=1)

    X_test = dataset_test[['Tweet']]
    y_test = dataset_test.drop(labels=['Tweet', 'set'] +
                                      [col for col in dataset.columns if col.startswith('fold')], axis=1)

    # lhs = ('care', 'fairness', 'loyalty', 'authority', 'purity')
    # rhs = ('harm', 'cheating', 'betrayal', 'subversion', 'degradation')
    #
    # for l, r in zip(lhs, rhs):
    #     for y in (y_train, y_test):
    #         y['{}_{}'.format(l, r)] = y[l] | y[r]
    #         y.drop(labels=[l, r], axis=1, inplace=True)

    return X_train, X_test, y_train, y_test


def get_tweet_folds(name):
    dataset = pd.read_csv('{}.csv'.format(name))
    dataset = dataset[dataset.set == 'train']
    folds_cols = [col for col in dataset.columns if col.startswith('fold')]
    for fold in folds_cols:
        v = dataset.loc[:, fold]
        yield np.where(v == 'train')[0], np.where(v == 'test')[0]


def get_dataset_cv(name):
    if name == 'bbc':
        return get_bbc_dataset_folds()
    return get_tweet_folds(name)


def get_bbc_dataset():
    """Extract a return the train and test data for the bbc corpus."""
    dataset = pd.read_csv('bbc_dataset.csv', index_col=0)
    dataset_train = dataset[dataset.set == 'train']
    dataset_test = dataset[dataset.set == 'test']

    X_train = dataset_train[['Utterance']]
    y_train = dataset_train.drop(['Utterance', 'set'], axis=1)

    X_test = dataset_test[['Utterance']]
    y_test = dataset_test.drop(['Utterance', 'set'], axis=1)
    return X_train, X_test, y_train, y_test


def get_bbc_dataset_folds():
    """Returns a generator for the train test split of the training data of the bbc_dataset"""
    X_train, _, _, _ = get_bbc_dataset()
    X_train_folds = pd.read_csv('bbc_dataset_folds.csv', index_col=0)
    data = X_train.join(X_train_folds)

    for fold in X_train_folds.columns:
        v = data.loc[:, fold]
        yield np.where(v == 'train')[0], np.where(v == 'test')[0]


def load_embeddings(dataset_name):
    if dataset_name == 'bbc':
        return load_bbc_elmo_embeddings()

    embeddings = pd.read_csv('{}_elmo_embeddings.csv'.format(dataset_name), index_col=0)
    dataset = pd.read_csv('{}.csv'.format(dataset_name), index_col=0)
    dataset_train = dataset[dataset.set == 'train']
    dataset_test = dataset[dataset.set == 'test']
    return embeddings.loc[dataset_train.index], embeddings.loc[dataset_test.index]


def load_bbc_elmo_embeddings():
    X_train = pd.read_csv('bbc_elmo_train_embeddings.csv', index_col=0)
    X_test = pd.read_csv('bbc_elmo_test_embeddings.csv', index_col=0)
    return X_train, X_test


_class_metrics = {
    'f1': lambda a, x: f1_score(x[0], x[1], average=a),
    'recall': lambda a, x: recall_score(x[0], x[1], average=a),
    'precision': lambda a, x: precision_score(x[0], x[1], average=a),
    'accuracy': lambda a, x: accuracy_score(x[0], x[1]),
    'zero_one_loss': lambda a, x: zero_one_loss(x[0], x[1]),
    'jaccard': lambda a, x: jaccard_similarity_score(x[0], x[1]),
}

_overall_metrics = {
    'f1': lambda x: f1_score(x[0], x[1], average='macro'),
    'recall': lambda x: recall_score(x[0], x[1], average='macro'),
    'precision': lambda x: precision_score(x[0], x[1], average='macro'),
    'accuracy': lambda x: accuracy_score(x[0], x[1]),
    'zero_one_loss': lambda x: zero_one_loss(x[0], x[1]),
    'jaccard': lambda x: jaccard_similarity_score(x[0], x[1]),
}


def calc_scores(y_true, y_pred, average='binary'):
    scores = pd.DataFrame(index=y_true.columns.tolist() + ['OVERALL'], columns=sorted(_class_metrics.keys()))
    # class scores
    for label in y_true.columns:
        yp = y_pred.loc[:, label]
        yt = y_true.loc[:, label]

        for n, scorer in _class_metrics.items():
            f = partial(scorer, a=average)
            scores.at[label, n] = f(x=(yt, yp))

    # overall scores
    for n, scorer in _overall_metrics.items():
        scores.at['OVERALL', n] = scorer((y_true, y_pred))
    return scores


def _int_or_str(x):
    try:
        return int(x)
    except ValueError as e:
        return str(x)


def encode_powerset_labels(y):
    """Encode the array y as a collection of concatenated label strings including
    each stance where the stance is 1. The label format is suitable for feeding to FastText."""
    return ['__label__' + '_'.join(map(str, y[i, :])) for i in range(y.shape[0])]


def decode_powerset_labels(y):
    """Decode the label and return a binary array. Assumes the labels are encoded
    the in the FastTest format: __label__x"""
    def decode_label(label):
        split_label = label.replace('__label__', '').split('_')
        return list(map(_int_or_str, split_label))
    return np.array(list(map(decode_label, y)))


def encode_binary_relevance_labels(arr):
    return np.apply_along_axis(lambda a: ['__label__{}'.format(i) for i in a],
                               axis=0, arr=arr)


def decode_binary_relevance_labels(y_pred):
    yp = [tuple(s.split(' ')) for s in y_pred.stdout.decode("utf-8").split('\n')]
    yp = list(filter(lambda x: len(x) == 2, yp))
    yp = np.array([(_int_or_str(l.split('__')[2]), float(p)) for (l, p) in yp])
    return np.array(yp)


def display_results(model_name, model, y_true, y_pred, dataset_name):
    print('{} Model Results'.format(model_name))
    print('cv_results_:')
    print(model.cv_results_)
    print('best_estimator_:')
    print(model.best_estimator_)
    print('best_params_:')
    print(model.best_params_)
    print('best_score_:')
    print(model.best_score_)

    if dataset_name == 'bbc':
        print('Test Set Results:')
        scores = calc_scores(y_true, y_pred)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(scores)
            print(scores.loc['OVERALL', :].T)
        return

    print('Tweet test score: {}'.format(_tweet_score(y_true.values, y_pred)))


def calc_label_corr(y):
    label_corr = pd.DataFrame(index=y.columns, columns=y.columns, data=0.0)
    for a, b in zip(*np.tril_indices(y.shape[1])):
        label_corr.iloc[a, b] = label_corr.iloc[b, a] = matthews_corrcoef(y.iloc[:, a], y.iloc[:, b])
    return label_corr
        
        
def plot_corr_matrix(y, threshold=0.15, title='Predicted', y_validate=None):
    plt.figure(figsize=(12, 9))
    plt.title('{} label correlation - threshold={}'.format(title, threshold))
    corr = calc_label_corr(y)
    corr[corr.abs() < threshold] = 0
    hm = sns.heatmap(corr, annot=True)
    return hm