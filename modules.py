"""
List of necessary modules for attention cluster.

@author: Juhan Bae
@reference:
"""

import tensorflow as tf
import base_module


class OneFcAttention(base_module.BaseAttentionModule):
    def __init__(self, feature_size, num_feature, num_cluster, do_shift=True):
        """ Initialize Two Fully-Connected Attention Module
        :param feature_size: int
            The size of a feature. e.g. 1024, 2048
        :param num_feature: int
            Total number of features to be considered when computing attention.
        :param num_cluster: int
            The number of clusters for attention.
        :param do_shift: bool
            True iff shifting operation is performed.
        """
        base_module.BaseAttentionModule.__init__(self, feature_size, num_feature, num_cluster, do_shift)

    def forward(self, inputs, **unused_inputs):
        """ Forward method for OneFcAttention.
        :param inputs: 3D tensor with dimension 'batch_size x num_feature x feature_size'
        :return: 2D tensor with dimension 'batch_size x num_feature'
        """
        reshaped_inputs = tf.reshape(inputs, [-1, self.feature_size])

        attention = tf.layers.dense(reshaped_inputs, self.num_cluster,
                                    activation=None,
                                    use_bias=True)

        float_cpy = tf.cast(tf.sqrt(self.feature_size), dtype=tf.float32)
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

        if self.do_shift:
            activation = tf.reshape(activation, [-1, self.feature_size])
            activation = self.shifting_operation(activation)

        final_activation = tf.reshape(activation, [-1, self.num_cluster * self.feature_size])

        return final_activation


class TwoFcAttention(base_module.BaseAttentionModule):
    def __init__(self, feature_size, num_feature, num_cluster, hidden_size, do_shift=True):
        """ Initialize Two Fully-Connected Attention Module
        :param feature_size: int
            The size of a feature. e.g. 1024, 2048
        :param num_feature: int
            Total number of features to be considered when computing attention.
        :param num_cluster: int
            The number of clusters for attention.
        :param do_shift: bool
            True iff shifting operation is performed.
        """
        base_module.BaseAttentionModule.__init__(self, feature_size, num_feature, num_cluster, do_shift)

    def forward(self, inputs, **unused_inputs):
        """ Forward method for TwoFcAttention.
        :param inputs: 3D tensor with dimension 'batch_size x num_feature x feature_size'
        :return: 2D tensor with dimension 'batch_size x num_feature'
        """
        reshaped_inputs = tf.reshape(inputs, [-1, self.feature_size])

        attention_temp = tf.layers.dense(reshaped_inputs, self.hidden_size,
                                         activation=tf.nn.tanh,
                                         use_bias=True,
                                         name="two_fc_attention_1")
        # -> '(batch_size * num_features) x hidden_size'

        attention = tf.layers.dense(attention_temp, self.num_cluster,
                                    activation=None,
                                    use_bias=True,
                                    name="two_fc_attention_2")
        # -> '(batch_size * num_features) x num_cluster'

        float_cpy = tf.cast(tf.sqrt(self.feature_size), dtype=tf.float32)
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

        activation = tf.reshape(activation, [-1, self.feature_size])

        if self.do_shift:
            activation = tf.reshape(activation, [-1, self.feature_size])
            activation = self.shifting_operation(activation)

        final_activation = tf.reshape(activation, [-1, self.num_cluster * self.feature_size])

        return final_activation
