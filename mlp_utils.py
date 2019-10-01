from collections import OrderedDict

import tensorflow as tf
from keras.losses import binary_crossentropy


def tf_cov(x):
    """Computes the covariance matrix of x"""
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    n = tf.cast(tf.shape(x)[1], tf.float32)
    return tf.matmul(tf.transpose(x - mean_x), x - mean_x) / n


def tf_corr(x):
    """Computes the correlation matrix of x, using the identity:
        https://en.wikipedia.org/wiki/Covariance_matrix#Relation_to_the_matrix_of_correlation_coefficients
    and setting NaN values to zero"""
    cov_x = tf_cov(x)
    d = tf.diag(1 / tf.sqrt(tf.matrix_diag_part(cov_x)))
    corr = tf.matmul(tf.matmul(d, cov_x), d)
    corr_no_nan = tf.where(tf.is_nan(corr), tf.zeros_like(corr), corr)
    return corr_no_nan


def correlation_delta(y_true, y_pred, threshold=0.1):
    """Computes a correlation delta from the Frobenius norm of the difference between the
    correlation matrices of y_true and y_pred 'up to' a given threshold, ie. all correlation
    values below threshold are set to zero beforehand."""
    y_true_corr = tf_corr(y_true)
    y_true_corr_thresholded = tf.where(tf.abs(y_true_corr) < threshold,
                                       tf.zeros_like(y_true_corr), y_true_corr)

    y_pred_corr = tf_corr(y_pred)
    y_pred_corr_thresholded = tf.where(tf.abs(y_pred_corr) < threshold,
                                       tf.zeros_like(y_pred_corr), y_pred_corr)

    y_true_corr_ut = tf.matrix_band_part(y_true_corr_thresholded, 0, -1)
    y_pred_corr_ut = tf.matrix_band_part(y_pred_corr_thresholded, 0, -1)

    return tf.norm(y_true_corr_ut - y_pred_corr_ut, ord='fro', axis=(0, 1))


def cld_tf(x, split=None):
    """Implements cross label dependency loss as defined in:
    https://arxiv.org/pdf/1707.00418.pdf for given y_true and y_pred"""
    y_true_tf = x[0]
    y_pred_tf = x[1]

    if split is None:
        y_pred_zero = tf.where(1 - y_true_tf)
        y_pred_one = tf.where(y_true_tf)
    else:
        y_pred_zero = tf.where(1 - y_true_tf[:split])
        y_pred_one = tf.where(y_true_tf[split:]) + split

    c = tf.meshgrid(y_pred_zero, y_pred_one, indexing='ij')
    d = tf.stack(c, axis=-1)
    e = tf.reshape(d, (-1, 2))
    f = tf.gather(y_pred_tf, e)
    g = tf.broadcast_to([1, -1], f[0].shape)
    h = tf.multiply(f, tf.cast(g, tf.float32))
    i = tf.exp(tf.reduce_sum(h, 1))
    j = tf.reduce_sum(i)
    return j / (tf.cast(tf.shape(y_pred_zero)[0], dtype=tf.float32) * tf.cast(tf.shape(y_pred_one)[0],
                                                                              dtype=tf.float32))


class CorrelationLoss(object):
    """Computes a correlation loss, which is the Frobenius norm of the difference in
    the label correlation matrices of the batch predicted labels, and either the
    correlation matrix of the batch true labels or that of the global true labels."""
    def __init__(self, alpha=0.0, threshold=0.0, use_y_true=False):
        self.alpha = alpha
        self.threshold = threshold
        self.use_y_true = use_y_true
        self.y_true = None

    def get_params(self):
        return OrderedDict(alpha=self.alpha, threshold=self.threshold)

    def set_y_true(self, y_true):
        self.y_true = y_true

    def __call__(self, y_true, y_pred):
        # in correlation loss, use batch y_true if global y_true not supplied
        yt = y_true if not self.use_y_true else tf.constant(self.y_true, dtype=tf.float32)
        corr_loss = correlation_delta(yt, y_pred, self.threshold)
        return binary_crossentropy(y_true, y_pred) + self.alpha * corr_loss

    def __str__(self):
        style = 'None' if self.y_true is None else 'not None'
        return 'CorrelationLoss(alpha={:.2f}, threshold={:.2f}, y_pred={}'.format(self.alpha, self.threshold, style)

    def __repr__(self):
        return self.__str__()


class CrossLabelDependencyLoss(object):
    """Computes the cross-label dependency loss between the batch true labels and the batch
    predicted labels"""
    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def get_params(self):
        return OrderedDict(alpha=self.alpha)

    def __call__(self, y_true, y_pred):
        elems = tf.stack([y_true, y_pred], 1)
        cld_loss = tf.reduce_sum(tf.map_fn(cld_tf, elems))
        return binary_crossentropy(y_true, y_pred) + self.alpha * cld_loss

    def __str__(self):
        return 'CrossLabelDependencyLoss(alpha={:.2f})'.format(self.alpha)

    def __repr__(self):
        return self.__str__()