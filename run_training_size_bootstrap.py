import click
import pickle
import numpy as np
from collections import defaultdict
from utils import reset_seeds, get_dataset, load_embeddings
from mlp_multilabel_wrapper import PowersetKerasWrapper, MultiOutputKerasWrapper
from mlp_utils import CrossLabelDependencyLoss


def get_random_sample(dataset_name='bbc', train_frac=0.25):
    # get model runner specific dataset
    _, _, y_train, y_test = get_dataset(dataset_name)
    X_train, X_test = load_embeddings(dataset_name)

    grps = y_train.apply(lambda v: ''.join(map(str, v)), axis=1).to_frame(0).groupby(0)[0]
    train_idx = grps.apply(lambda g: g.sample(frac=train_frac)).index.get_level_values(1)

    X_train_sample = X_train.loc[train_idx, :]
    y_train_sample = y_train.loc[train_idx, :]
    return X_train_sample, X_test, y_train_sample, y_test


def _get_label_set(y):
    return set(y.apply(lambda v: ''.join(map(str, v)), axis=1).values)


@click.command()
@click.option('--n-samples', default=10)
@click.option('--dataset-name', default='moral-dataset-MeToo')
def run(n_samples, dataset_name):
    mlp_cld_bootstrap_results = defaultdict(lambda: defaultdict(list))
    mlp_powerset_bootstrap_results = defaultdict(lambda: defaultdict(list))
    mlp_labels_bootstrap_results = defaultdict(lambda: defaultdict(list))

    reset_seeds()
    for i in range(n_samples):
        print('Running bootstrap sample: {}'.format(i + 1))
        for f in np.arange(0.1, 1.1, 0.1):
            X_train, X_test, y_train, y_test = get_random_sample(dataset_name, train_frac=f)

            print('Training set size: {}'.format(X_train.shape))
            print('Test set size: {}'.format(X_test.shape))

            mlp_powerset_model = PowersetKerasWrapper(columns=y_train.columns)
            mlp_powerset_model.fit(X_train.values, y_train.values)
            y_pred_mlp = mlp_powerset_model.predict(X_test.values)
            mlp_powerset_bootstrap_results[i][f].append(y_pred_mlp)

            cld_loss = CrossLabelDependencyLoss(alpha=0.2)
            mlp_cld_model = MultiOutputKerasWrapper(columns=y_train.columns, loss=cld_loss)
            mlp_cld_model.fit(X_train.values, y_train.values)
            y_pred_cld = mlp_cld_model.predict(X_test.values)
            mlp_cld_bootstrap_results[i][f].append(y_pred_cld)

            mlp_labels_bootstrap_results[i][f].append((_get_label_set(y_train), _get_label_set(y_test)))

        with open('training_size_bootstrap_{}.pkl'.format(dataset_name), 'wb') as f:
            pickle.dump({'cld': dict(mlp_cld_bootstrap_results),
                         'powerset': dict(mlp_powerset_bootstrap_results),
                         'labels': dict(mlp_labels_bootstrap_results)}, f)


if __name__ == '__main__':
    run()
