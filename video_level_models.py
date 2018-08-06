# Copyright 2018 Juhan Bae All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
# noinspection PyUnresolvedReferences
import pathmagic
from tensorflow import flags
import tensorflow as tf
import tensorflow.contrib.slim as slim
import models
import math

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")
flags.DEFINE_float(
    "moe_l2", 1e-8,
    "L2 penalty for MoeModel.")


class MoeModel(models.BaseModel):
    """ A softmax over a mixture of logistic models (with L2 regularization). """

    def create_model(self,
                     model_input,
                     vocab_size,
                     is_training,
                     num_mixtures=None,
                     l2_penalty=1e-8,
                     filter_size=2,
                     ff_dropout_rate=0.9,
                     **unused_params):
        """Creates a Mixture of (Logistic) Experts model.
         It also includes the possibility of gating the probabilities
         The model consists of a per-class softmax distribution over a
         configurable number of logistic classifiers. One of the classifiers in the
         mixture is not trained, and always predicts 0.
        Args:
          model_input: 'batch_size' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.
          is_training: Is this the training phase ?
          num_mixtures: The number of mixtures (excluding a dummy 'expert' that
            always predicts the non-existence of an entity).
          l2_penalty: How much to penalize the squared magnitudes of parameter
            values.
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes.
        """
        num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
        l2_penalty = FLAGS.moe_l2

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")

        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities = tf.reshape(probabilities_by_class_and_batch,
                                   [-1, vocab_size])

        # Deep context gating.
        gating_weights1 = tf.layers.dense(probabilities, vocab_size * filter_size,
                                          use_bias=False, activation=tf.nn.relu)
        gating_weights1 = tf.layers.batch_normalization(gating_weights1, training=is_training)
        if is_training:
            gating_weights1 = tf.nn.dropout(gating_weights1, ff_dropout_rate)

        gating_weights2 = tf.layers.dense(gating_weights1, vocab_size,
                                          use_bias=False, activation=None)
        gating_weights2 = tf.layers.batch_normalization(gating_weights2, training=is_training)
        gating_weights2 = tf.sigmoid(gating_weights2)
        activation = tf.multiply(probabilities, gating_weights2)

        return {"predictions": activation}
