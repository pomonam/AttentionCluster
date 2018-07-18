import tensorflow as tf


class BaseAttentionModule:
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
        self.feature_size = feature_size
        self.num_feature = num_feature
        self.num_cluster = num_cluster
        self.hidden_size = hidden_size
        self.do_shift = do_shift

    def forward(self, inputs, **unused_inputs):
        raise NotImplementedError("BaseModule is an abstract class for other modules.")

    def shifting_operation(self, inputs, scope_id=None):
        """ Shifting operation for attention.
        :param inputs: 2D Tensor of size '(batch_size * num_cluster) x feature_size'
        :param scope_id: String
        :return: 2D Tensor of size 'batch_size x feature_size'
        """
        alpha = tf.get_variable("alpha{}".format(str(scope_id)),
                                [1],
                                initializer=tf.constant_initializer(1))
        beta = tf.get_variable("beta{}".format(str(scope_id)),
                               [1],
                               initializer=tf.constant_initializer(0))
        activation = alpha * inputs
        activation = activation + beta
        normalized_activation = tf.nn.l2_normalize(activation, 1)
        float_cpy = tf.cast(tf.sqrt(self.num_cluster), dtype=tf.float32)
        final_activation = tf.divide(normalized_activation, float_cpy)

        return final_activation

