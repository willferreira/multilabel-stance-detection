import os
os.environ['PYTHONHASHSEED'] = str(0)

import pickle
import tensorflow as tf
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import click
from collections import defaultdict
import pandas as pd
import py

from utils import reset_seeds, get_dataset, load_embeddings
from run_cv import models


def get_random_sample(dataset_name, frac=0.8):
    # get model runner specific dataset
    _, _, y_train, y_test = get_dataset(dataset_name)
    X_train, X_test = load_embeddings(dataset_name)

    # combine train and test and then sample
    X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)

    grps = y.apply(lambda v: ''.join(map(str, v)), axis=1).to_frame(0).groupby(0)[0]
    train_idx = grps.apply(lambda g: g.sample(frac=frac)).index.get_level_values(1)

    X_train = X.loc[train_idx, :]
    y_train = y.loc[train_idx, :]
    test_idx = list(set(X.index).difference(X_train.index))
    X_test = X.loc[test_idx, :]
    y_test = y.loc[test_idx, :]
    return X_train, X_test, y_train, y_test


@click.command()
@click.option('--n-samples', default=30)
@click.option('--sample-frac', default=0.7)
@click.option('--dataset-name', default='moral-dataset-MeToo', type=click.Choice(['bbc', 'tweets-DT_HC',
                                                                           'tweets-DT_TC', 'tweets-HC_BS',
                                                                           'moral-dataset-BLM', 'moral-dataset-ALM',
                                                                           'moral-dataset-Election', 'moral-dataset-Baltimore',
                                                                                'moral-dataset-MeToo',
                                                                                'moral-dataset-Davidson',
                                                                                  ]))
def run(n_samples, sample_frac, dataset_name):
    path = py.path.local('results/')
    files = path.listdir(fil='cv_results*.pkl')

    # keep all the results we're interested in
    cv_results = {}
    for file in files:
        with open(file.strpath, 'rb') as f:
            results = pickle.load(f)
        model_name = results['model_name']
        if results['model_name'].startswith('mlp') and results['dataset_name'] == dataset_name:
            cv_results[model_name] = results
    print(cv_results.keys())
    bootstrap_results = defaultdict(list)
    reset_seeds()
    for i in range(n_samples):
        print('Running bootstrap sample: {}'.format(i+1))
        X_train, X_test, y_train, y_test = get_random_sample(dataset_name, frac=sample_frac)

        print('Training set size: {}'.format(X_train.shape))
        print('Test set size: {}'.format(X_test.shape))

        for m_name, results in cv_results.items():
            print('Estimating model: {}'.format(m_name))
            print('Best params: {}'.format(results['best_params']))
            _, clz, _ = models[m_name]
            params = dict(columns=y_train.columns, epochs=50, batch_size=32, verbose=True)
            params.update(results['best_params'])
            model = clz(**params)
            model.fit(X_train.values, y_train.values)
            y_pred = model.predict(X_test.values)
            y_pred = pd.DataFrame(data=y_pred)
            y_pred.columns = model.columns
            bootstrap_results[m_name].append((y_pred, y_test))
        # partial save in case it quits
        with open('results/bootstrap_results_{}.pkl'.format(dataset_name), 'wb') as f:
            pickle.dump(bootstrap_results, f)


if __name__ == '__main__':
    run()