import numpy as np
import tensorflow as tf


def xavier_weight(shape, dtype):
    W = tf.random_uniform(shape=shape, \
                          minval=-np.sqrt(6. / (shape[0] + shape[1])), \
                          maxval=np.sqrt(6. / (shape[0] + shape[1])), \
                          dtype=dtype)
    return W


class XavierInitializer(object):

    def __init__(self, dtype=tf.float32):
        self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype

        return xavier_weight(shape, dtype)


def init_weight(shape, name, l2_reg=0.):
    scope = tf.get_variable_scope()
    if not scope.reuse:
        initializer = XavierInitializer()
        v = tf.get_variable(name, shape, tf.float32, initializer,
                            tf.contrib.layers.l2_regularizer(l2_reg))
    else:
        v = tf.get_variable(name)
    return v


def init_bias(size, name, l2_reg=0.):
    scope = tf.get_variable_scope()

    if not scope.reuse:
        v = tf.get_variable(name, (size,), tf.float32, \
                            tf.zeros_initializer(), \
                            tf.contrib.layers.l2_regularizer(l2_reg))
    else:
        v = tf.get_variable(name)
    return v


def init_state(size, name, init_fac):
    scope = tf.get_variable_scope()

    if not scope.reuse:
        v = tf.get_variable(name, (size,), tf.float32, \
                            tf.random_normal_initializer(stddev=np.sqrt(init_fac)))
    else:
        v = tf.get_variable(name)
    return v
