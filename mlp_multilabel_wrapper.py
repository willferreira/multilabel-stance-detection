import functools as ft

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.models import Model
from keras.utils import np_utils
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, label_binarize

from utils import encode_powerset_labels, decode_powerset_labels
from mlp_utils import cld_tf, CorrelationLoss


def _get_multilabel_class_weight(y):
    """Weight classes by inverse of frequency - less frequent classes weighted higher"""
    w = np.sum(y) / np.sum(y, axis=0)
    return w / w.sum()


class BaseWrapperMixin(object):
    def build_model(self, pred_shape, output_activation='sigmoid', input_shape=1024, y=None):
        input = Input(shape=(input_shape,), dtype=tf.float32)

        hidden = Dropout(self.dropout)(Dense(128, activation='relu')(input))

        outputs = Dense(pred_shape, activation=output_activation)(hidden)
        model = Model(inputs=input, outputs=outputs)
        # hack to pass the training batch y labels to the loss function
        if isinstance(self.loss, CorrelationLoss):
            self.loss.set_y_true(y)
        model.compile(optimizer='adam', loss=self.loss)

        if self.verbose:
            print(model.summary())

        return model


class MultiOutputKerasWrapper(BaseEstimator, BaseWrapperMixin, ClassifierMixin):
    def __init__(self, columns, epochs=50, batch_size=32, loss='binary_crossentropy', dropout=0.5, verbose=True):
        self.columns = columns
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.dropout = dropout
        self.verbose = verbose
        self.model = None

    def fit(self, X, y, sample_weight=None):
        class_weight = _get_multilabel_class_weight(y)
        self.model = self.build_model(pred_shape=y.shape[1], y=y)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose,
                       class_weight=class_weight)
        return self

    def predict(self, X):
        y_pred_proba = self.model.predict(X)
        return pd.DataFrame(data=(y_pred_proba > 0.5), columns=self.columns).astype(int)


class PowersetKerasWrapper(BaseEstimator, BaseWrapperMixin, ClassifierMixin):
    def __init__(self, columns, epochs=50, batch_size=32, loss='categorical_crossentropy', dropout=0.5, verbose=True):
        self.columns = columns
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.dropout = dropout
        self.verbose = verbose
        self.model = None
        self.label_encoder = None

    def _do_categorical_label_encoding(self, y):
        self.label_encoder = LabelEncoder()
        return np_utils.to_categorical(
            self.label_encoder.fit_transform(
                encode_powerset_labels(y)
            )
        )

    def fit(self, X, y, sample_weight=None):
        y_enc = self._do_categorical_label_encoding(y)
        class_weight = _get_multilabel_class_weight(y_enc)

        self.model = self.build_model(output_activation='softmax', pred_shape=self.label_encoder.classes_.shape[0])
        self.model.fit(X, y_enc, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose,
                       class_weight=class_weight)
        return self

    def predict(self, X):
        y_pred_proba = self.model.predict(X)
        y_pred_labels = self.label_encoder.inverse_transform(np.argmax(y_pred_proba, axis=1))
        return pd.DataFrame(data=decode_powerset_labels(y_pred_labels),
                            columns=self.columns)


class MultitaskKerasWrapper(BaseEstimator, ClassifierMixin):
    """Multi-task model - essentially the same as the Multi-label version except there is a shared hidden
        layer and then one output per stance, each with its own hidden layer."""

    def __init__(self, columns, epochs=50, batch_size=32, dropout=0.5, verbose=True):
        self.columns = columns
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.dropout = dropout
        self.model = None

    def build_model(self, pred_shape, input_shape=1024):
        input = Input(shape=(input_shape,), dtype=tf.float32)
        shared_layer = Dense(80, activation=None)(input)
        shared_layer = BatchNormalization()(shared_layer)
        shared_layer = Activation(activation='relu')(shared_layer)
        shared_dropout = Dropout(self.dropout)(shared_layer)

        # separate hidden layers, one for each stance label
        stance_layers = [
            Dropout(self.dropout)(Dense(64, activation='relu')(shared_dropout))
            for _ in range(pred_shape)
        ]

        # separate output layers, one for each class
        outputs = [Dense(1, activation='sigmoid')(sl) for sl in stance_layers]
        model = Model(inputs=input, outputs=outputs)
        model.compile(optimizer='adam', loss=['binary_crossentropy'] * pred_shape)

        if self.verbose:
            print(model.summary())

        return model

    def fit(self, X, y, sample_weight=None):
        pred_shape = y.shape[1]
        self.model = self.build_model(pred_shape=pred_shape)
        y_ = [y[:, i] for i in range(y.shape[1])]
        self.model.fit(X, y_, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

    def predict(self, X):
        y_pred_proba = np.array(self.model.predict(X)).squeeze().T
        return pd.DataFrame(data=(y_pred_proba > 0.5), columns=self.columns).astype(int)


class MultilabelMultioutputKerasWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, columns, epochs=50, batch_size=32, dropout=0.5, verbose=True, alpha_left=0.0, alpha_right=0.0):
        self.columns = columns
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.verbose = verbose
        self.alpha_left = alpha_left
        self.alpha_right = alpha_right
        self.model = None

    def _cld_loss(self, outputs, y_true_target_1, y_true_target_2):
        # categorical cross-entropy loss
        y_true = tf.concat([y_true_target_1, y_true_target_2], 1)
        y_pred = tf.concat(outputs, 1)
        elems_left = tf.stack([y_true, y_pred], 1)
        cld_loss_left = tf.reduce_sum(tf.map_fn(
            ft.partial(cld_tf, split=y_true_target_1.shape[1]), elems_left))

        y_true = tf.concat([y_true_target_2, y_true_target_1], 1)
        y_pred = tf.concat(outputs[::-1], 1)
        elems_right = tf.stack([y_true, y_pred], 1)
        cld_loss_right = tf.reduce_sum(tf.map_fn(
            ft.partial(cld_tf, split=y_true_target_1.shape[1]), elems_right))

        return self.alpha_left * cld_loss_left + self.alpha_right * cld_loss_right

    def build_model(self, pred_shape, input_shape=1024):
        embedding_input = Input(shape=(input_shape,), dtype=tf.float32)
        y_true_target_1 = Input(shape=(3,), dtype=tf.float32)
        y_true_target_2 = Input(shape=(3,), dtype=tf.float32)

        hidden = Dropout(self.dropout)(Dense(128, activation='relu')(embedding_input))

        # separate output layers, one for each target
        outputs = [Dense(3, activation='softmax')(hidden) for _ in range(2)]
        model = Model(inputs=[embedding_input, y_true_target_1, y_true_target_2], outputs=outputs)
        model.add_loss(self._cld_loss(outputs, y_true_target_1, y_true_target_2))
        model.compile(optimizer='adam', loss=['categorical_crossentropy']*2)

        if self.verbose:
            print(model.summary())

        return model

    stances = ['FAVOR', 'AGAINST', 'NONE']

    def fit(self, X, y, sample_weight=None):
        pred_shape = y.shape[1]
        self.model = self.build_model(pred_shape=pred_shape)
        # include true outputs in input, to be able to compute cld loss
        y_ = [label_binarize(y[:, i], classes=self.stances) for i in range(y.shape[1])]
        X_ = [X] + y_
        self.model.fit(X_, y_, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

    def predict(self, X):
        # dummy inputs
        X_ = [X] + [np.zeros((X.shape[0], 3)), np.zeros((X.shape[0], 3))]
        y_pred_target_1 = [self.stances[i] for i in np.argmax(self.model.predict(X_)[0], axis=1)]
        y_pred_target_2 = [self.stances[i] for i in np.argmax(self.model.predict(X_)[1], axis=1)]
        return pd.DataFrame(data={self.columns[0]: y_pred_target_1, self.columns[1]: y_pred_target_2})

