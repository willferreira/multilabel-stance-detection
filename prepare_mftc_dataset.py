import json
import click
import itertools as it

from sklearn.model_selection import KFold
from collections import Counter
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from nltk.tokenize.nist import NISTTokenizer


def get_corpus_annotations(c):
    with open('MFTC_V3_Text.json', 'rb') as f:
        mftc = json.load(f)
    data = [d for d in mftc if d['Corpus'] == c].pop()
    return data


all_labels = ('authority', 'betrayal', 'care', 'cheating', 'degradation',
              'fairness', 'harm', 'loyalty', 'non-moral', 'purity', 'subversion')


def process_annotations(data):
    annots = []
    texts = []
    for x in data['Tweets']:
        tweet_id = x['tweet_id']
        ax = x['annotations']
        num_annotators = len(ax)
        c = Counter(it.chain(*[v['annotation'].split(',') for v in ax]))
        labels = [a for (a, n) in c.items() if n >= np.ceil(num_annotators / 2)]
        if len(labels) == 0 and 'non-moral' in c.keys():
            labels = ['non-moral']
        if len(labels) > 1 and 'non-moral' in labels:
            labels.remove('non-moral')
        annots.append((tweet_id, labels))
        texts.append((tweet_id, x['tweet_text']))
    df_labels = pd.DataFrame(index=[id for (id, _) in annots], columns=all_labels, data=0)
    df_text = pd.DataFrame(index=[id for (id, _) in texts], columns=['Tweet'])
    for id, an in annots:
        df_labels.at[id, an] = 1
    for id, txt in texts:
        df_text.at[id, 'Tweet'] = txt
    df_labels.drop(labels='non-moral', axis=1, inplace=True)
    df_labels.drop(labels=df_labels[df_labels.sum(axis=1) == 0].index, inplace=True)
    return df_labels.astype(int).join(df_text)


def get_train_test_split(corpus, annotations, n_splits=5, train_test_split=0.8, cutoff=3):
    annotations_no_tweet = annotations.drop(labels='Tweet', axis=1)
    grps = annotations_no_tweet.apply(lambda v: ''.join(map(str, v)), axis=1).to_frame(0).groupby(0)[0]
    test_idx = grps.apply(lambda g: g.sample(frac=1-train_test_split)).index.get_level_values(1)
    train_idx = set(annotations_no_tweet.index).difference(test_idx)

    annotations.at[train_idx, 'set'] = 'train'
    annotations.at[test_idx, 'set'] = 'test'

    train_grps = annotations_no_tweet.loc[train_idx, :].apply(lambda v: ''.join(map(str, v)), axis=1) \
        .to_frame(0).groupby(0)[0]
    for i in range(n_splits):
        fold_test_idx = train_grps.apply(lambda g: g.sample(frac=1/n_splits)).index.get_level_values(1)
        fold_train_idx = set(train_idx).difference(fold_test_idx)
        fold_id = 'fold_{}'.format(i + 1)
        annotations[fold_id] = None
        annotations.loc[fold_train_idx, fold_id] = 'train'
        annotations.loc[fold_test_idx, fold_id] = 'test'

    tokenizer = NISTTokenizer()
    annotations.Tweet = annotations.Tweet.apply(
        lambda x: ' '.join(tokenizer.tokenize(x, lowercase=True)))

    annotations.to_csv('moral-dataset-{}.csv'.format(corpus))

    elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=False)

    embeds = []
    for i in range(0, annotations.shape[0] // 100 + 1):
        print('Computing embeddings for [{} .. {})'.format(i*100, (i+1)*100))
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            tweets = annotations[['Tweet']].iloc[(i*100):(i+1)*100, :]
            if tweets.shape[0] > 0:
                elmo_tweet_embeddings = session.run(elmo(tf.squeeze(tf.cast(tweets.values, tf.string)),
                                                         signature='default', as_dict=True)['default'])
                embeds.append(pd.DataFrame(index=tweets.index, data=elmo_tweet_embeddings))
    all_embeds = pd.concat(embeds, 0)
    all_embeds.to_csv('moral-dataset-{}_elmo_embeddings.csv'.format(corpus))


def __old__get_train_test_split(corpus, annotations, n_splits=5, train_test_split=0.8, cutoff=3):
    annotations_no_tweet = annotations.drop(labels='Tweet', axis=1)
    to_drop = annotations_no_tweet[annotations_no_tweet.sum(axis=1) > cutoff].index
    annotations.drop(labels=to_drop, axis=0, inplace=True)

    train_idx = set()
    test_idx = set()

    for i in range(1, cutoff+1, 1):
        an = annotations[annotations.sum(axis=1) == i]
        train_sample = an.sample(frac=train_test_split)
        train_idx.update(train_sample.index)
        test_idx.update(set(an.index).difference(train_sample.index))

    annotations.at[train_idx, 'set'] = 'train'
    annotations.at[test_idx, 'set'] = 'test'

    kf = KFold(n_splits=n_splits)
    train_data = annotations[annotations.set == 'train']
    for i, (train_idx, test_idx) in enumerate(kf.split(train_data)):
        fold_id = 'fold_{}'.format(i + 1)
        annotations[fold_id] = None
        col_id = annotations.columns.get_loc(fold_id)
        annotations.iloc[train_idx, col_id] = 'train'
        annotations.iloc[test_idx, col_id] = 'test'

    tokenizer = NISTTokenizer()
    annotations.Tweet = annotations.Tweet.apply(
        lambda x: ' '.join(tokenizer.tokenize(x, lowercase=True)))

    annotations.to_csv('moral-dataset-{}.csv'.format(corpus))

    elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=False)

    embeds = []
    for i in range(0, annotations.shape[0] // 100 + 1):
        print('Computing embeddings for [{} .. {})'.format(i*100, (i+1)*100))
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            tweets = annotations[['Tweet']].iloc[(i*100):(i+1)*100, :]
            if tweets.shape[0] > 0:
                elmo_tweet_embeddings = session.run(elmo(tf.squeeze(tf.cast(tweets.values, tf.string)),
                                                         signature='default', as_dict=True)['default'])
                embeds.append(pd.DataFrame(index=tweets.index, data=elmo_tweet_embeddings))
    all_embeds = pd.concat(embeds, 0)
    all_embeds.to_csv('moral-dataset-{}_elmo_embeddings.csv'.format(corpus))


@click.command()
@click.option('--corpus', default='MeToo')
@click.option('--n-splits', default=5)
def run(corpus, n_splits):
    data = get_corpus_annotations(corpus)
    annotations = process_annotations(data)
    annotations.to_csv('moral_stance_annotations_{}.csv'.format(corpus))

    get_train_test_split(corpus, annotations, n_splits=n_splits)


if __name__ == '__main__':
    run()
