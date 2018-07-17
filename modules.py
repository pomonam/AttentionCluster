"""
List of necessary modules for attention cluster.

@author: Juhan Bae
@reference:
"""

import tensorflow as tf
import math


class OneFcAttention:
    def __init__(self, feature_size, num_feature, num_cluster, do_shift=True):
        """ Initialize One Fully-Connected Attention Module
        :param feature_size: int
            The size of a feature. e.g. 1024, 2048
        :param num_feature: int
            Total number of features to be considered when computing attention.
        :param num_cluster: int
            The number of clusters for attention.
        :param do_shift: bool
            True iff shifting operation is performed.
        """
        self.feature_size = feature_size
        self.num_feature = num_feature
        self.num_cluster = num_cluster
        self.do_shift = do_shift

    def forward(self, inputs):
        """ Forward method for OneFcAttention.
        :param inputs: 3D tensor with dimension 'batch_size x num_feature x feature_size'
        :return: 2D tensor with dimension 'batch_size x num_feature'
        """
        reshaped_inputs = tf.reshape(inputs, [-1, self.feature_size])
        attention_weights = tf.get_variable("one_fc_attention",
                                            [self.feature_size, self.num_cluster],
                                            initializer=tf.contrib.layers.xavier_initializer())
        attention = tf.matmul(reshaped_inputs, attention_weights)
        float_cpy = tf.cast(math.sqrt(self.feature_size), dtype=tf.float32)
        attention = tf.divide(attention, float_cpy)
        # -> '(batch_size * num_feature) x num_cluster'
        attention = tf.reshape(attention, [-1, self.num_feature, self.num_cluster])
        attention = tf.nn.softmax(attention, dim=1)
        # -> 'batch_size x num_feature x num_cluster'

        reshaped_inputs = tf.reshape(inputs, [-1, self.num_feature, self.feature_size])
        activation = tf.transpose(attention, perm=[0, 2, 1])
        # -> 'batch_size x num_cluster x num_feature'
        activation = tf.matmul(activation, reshaped_inputs)
        # -> 'batch_size x num_cluster x feature_size'

        reshaped_activation = tf.reshape(activation, [-1, self.feature_size])

        if self.do_shift:
            alpha = tf.get_variable("alpha",
                                    [1],
                                    initializer=tf.constant_initializer(1))
            beta = tf.get_variable("beta",
                                   [1],
                                   initializer=tf.constant_initializer(0.01))
            reshaped_activation = alpha * reshaped_activation
            reshaped_activation = reshaped_activation + beta
            reshaped_activation = tf.nn.l2_normalize(reshaped_activation, 1)
            float_cpy = tf.cast(math.sqrt(self.num_cluster), dtype=tf.float32)
            reshaped_activation = tf.divide(reshaped_activation, float_cpy)

        final_activation = tf.reshape(reshaped_activation, [-1, self.num_cluster * self.feature_size])

        return final_activation
