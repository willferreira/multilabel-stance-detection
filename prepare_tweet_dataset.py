import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.model_selection import KFold
from nltk.tokenize.nist import NISTTokenizer


def get_initials(s, t):
    parts1 = s.split(' ')
    parts2 = t.split(' ')
    return '{}{}_{}{}'.format(parts1[0][0], parts1[1][0], parts2[0][0], parts2[1][0])


def process_tweets(pair_id, target_pair, n_splits):
    target_pair['set'] = None
    target_pair['set'][target_pair['Test/Train/Dev'].isin(['Train', 'Dev'])] = 'train'
    target_pair['set'][pd.isnull(target_pair['set'])] = 'test'

    kf = KFold(n_splits=n_splits)
    target_pair_train = target_pair[target_pair.set == 'train']
    for i, (train_idx, test_idx) in enumerate(kf.split(target_pair_train)):
        fold_id = 'fold_{}'.format(i+1)
        target_pair[fold_id] = None
        target_pair[fold_id].iloc[train_idx] = 'train'
        target_pair[fold_id].iloc[test_idx] = 'test'

    tokenizer = NISTTokenizer()
    target_pair.Tweet = target_pair.Tweet.apply(
        lambda x: ' '.join(tokenizer.tokenize(x, lowercase=True)))

    target_pair.rename(columns={'Stance 1': 'Target 1',
                                'Stance 2': 'Target 2'}, inplace=True)
    target_pair.to_csv('tweets-{}.csv'.format(pair_id))

    elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=False)

    embeds = []
    print('There are {} tweets in pair: '.format(target_pair.shape[0], pair_id))
    for i in range(0, target_pair.shape[0] // 100 + 1):
        print('Computing embeddings for [{} .. {})'.format(i*100, (i+1)*100))
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            tweets = target_pair[['Tweet']].iloc[(i*100):(i+1)*100, :]
            if tweets.shape[0] > 0:
                elmo_tweet_embeddings = session.run(elmo(tf.squeeze(tf.cast(tweets.values, tf.string)),
                                                         signature='default', as_dict=True)['default'])
                embeds.append(pd.DataFrame(index=tweets.index, data=elmo_tweet_embeddings))
    all_embeds = pd.concat(embeds, 0)
    print('There are {} embeddings in pair: '.format(all_embeds.shape[0], pair_id))
    all_embeds.to_csv('tweets-{}_elmo_embeddings.csv'.format(pair_id))


def run(n_splits=5):
    tweets = pd.read_csv('all_data_tweet_text.csv')

    # 1. break up data into target-pair groups
    # 2. assign to train/test sets
    # 3. create 5 folds for cv
    for k, v in tweets.groupby(['Target 1', 'Target 2']).groups.items():
        print('Processing tweets for target-pair: {}'.format(k))
        group_data = tweets.loc[v, :]
        target_pair = group_data.drop(['Target 1', 'Target 2'], axis=1).copy()
        process_tweets(get_initials(*k), target_pair, n_splits)


if __name__ == '__main__':
    run()