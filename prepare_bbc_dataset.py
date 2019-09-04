"""
Run this script to generate the version of the Brexit Blog Corpus dataset
used in this study.

Input: brexit_blog_corpus.xlsx
Outputs: bbc_dataset.csv

The script does the following:

1. cleans the input data set to correct some inconsistent stance spelling
2. removes duplicate utterances
3. tokenizes the utterance text
4. binarizes the response
5. constructs train, test and validate sets
6. saves the dataset into a new file bbc_dataset.csv with a train/test/validate
   indicator
7. Uses ELMO to convert the text into 1024 dimensional word embeddings

The numpy seed is set to 1 at the beginning to ensure the dataset is reproducable.
"""
import pandas as pd
import numpy as np

import tensorflow_hub as hub
import tensorflow as tf

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from nltk.tokenize.nist import NISTTokenizer

from utils import reset_seeds


def run(n_splits=5):
    """
    Prepare the Brexit Blog Corpus data set for analysis
    :param n_splits: int, the number of train/test splits to degenerate, default=5
    :return:
    """
    print('Reading and processing the xlsx file...', end='')
    brexit_blog_corpus = pd.read_excel('brexit_blog_corpus.xlsx')

    # fix up some typos
    brexit_blog_corpus.replace('concession/contrarines', np.nan, inplace=True)
    brexit_blog_corpus.replace('hypotheticallity', 'hypotheticality', inplace=True)

    # unfortunately, quite a few utterances are duplicates :(
    clean_dataset = brexit_blog_corpus.drop_duplicates(subset='Utterance')

    stance_columns = ['Stance category', 'second stance category', 'third', 'fourth', 'fifth']

    clean_dataset = clean_dataset[['Utterance ID No', 'Utterance']
                                  + stance_columns].set_index('Utterance ID No')

    # extract the stance categories and do some cleaning
    stance_categories = set(clean_dataset[stance_columns].values.flatten())
    stance_categories.discard(np.nan)
    stance_categories = sorted(list(stance_categories))
    stance_categories = [w.replace(' ', '-').replace('/', '-') for w in stance_categories]

    # one-hot encode the assigned stance labels
    mlb = MultiLabelBinarizer()
    k_hot_encoded_stances = mlb.fit_transform([x[~pd.isnull(x)]
                                               for x in clean_dataset[stance_columns].values])
    k_hot_encoded_stances = pd.DataFrame(index=clean_dataset.index, data=k_hot_encoded_stances,
                                         columns=list(mlb.classes_))
    k_hot_encoded_stances.columns = stance_categories

    # join the one-hot encoded labels and utterances back together again
    clean_dataset_one_hot = clean_dataset[['Utterance', 'Stance category']] \
        .join(k_hot_encoded_stances)
    print('done.')

    print('Tokenising the utterances...', end='')
    # tokenize the Utterance
    tokenizer = NISTTokenizer()
    clean_dataset_one_hot.Utterance = clean_dataset_one_hot.Utterance.apply(
        lambda x: ' '.join(tokenizer.tokenize(x, lowercase=True)))
    print('done.')

    print('Constructing train/test split and saving to disk...', end='')
    # split the data into train and test sets in the ratio 80:20
    stance_columns = set(clean_dataset_one_hot.columns).difference(['Utterance', 'Stance category'])
    stance_columns = sorted(list(stance_columns))

    # first split the data in two to get train and test sets
    reset_seeds()

    X_train, X_test, y_train, y_test = \
        train_test_split(clean_dataset_one_hot['Utterance'],
                         clean_dataset_one_hot[stance_columns],
                         test_size=0.2,
                         stratify=clean_dataset_one_hot['Stance category'])

    y_train['set'] = 'train'
    y_test['set'] = 'test'

    dataset = pd.concat([
        pd.DataFrame(data={'Utterance': X_train}).join(y_train),
        pd.DataFrame(data={'Utterance': X_test}).join(y_test)
    ], axis=0)

    dataset.to_csv('bbc_dataset.csv')
    print('done.')

    print('Constructing the cv folds and saving to disk...', end='')
    X_train_folds = pd.DataFrame(index=X_train.index,
                                 columns=['fold_{}'.format(i) for i in range(1, n_splits+1)])
    skf = StratifiedKFold(n_splits=n_splits)
    y = clean_dataset_one_hot.loc[y_train.index, 'Stance category']
    for i, (train_idx, test_idx) in enumerate(skf.split(np.zeros(X_train.shape[0]), y)):
        X_train_folds.iloc[train_idx, i] = 'train'
        X_train_folds.iloc[test_idx, i] = 'test'

    X_train_folds.to_csv('bbc_dataset_folds.csv')
    print('done.')

    print('Pre-computing the ELMO embeddings and saving to disk...', end='')

    elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=False)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        elmo_train_embeddings = session.run(elmo(tf.squeeze(tf.cast(X_train.values, tf.string)),
                                                 signature='default', as_dict=True)['default'])
        elmo_train_embeddings = pd.DataFrame(index=X_train.index, data=elmo_train_embeddings)
        elmo_train_embeddings.to_csv('bbc_elmo_train_embeddings.csv')

        elmo_test_embeddings = session.run(elmo(tf.squeeze(tf.cast(X_test.values, tf.string)),
                                                signature='default', as_dict=True)['default'])
        elmo_test_embeddings = pd.DataFrame(index=X_test.index, data=elmo_test_embeddings)
        elmo_test_embeddings.to_csv('bbc_elmo_test_embeddings.csv')

    print('done.')


if __name__ == '__main__':
    run()