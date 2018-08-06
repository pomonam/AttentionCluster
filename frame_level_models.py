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

"""Contains a collection of models which operate on variable-length sequences."""
# noinspection PyUnresolvedReferences
import pathmagic
from tensorflow import flags
import tensorflow as tf
import video_level_models
import models
import modules


flags.DEFINE_integer("video_cluster_size", 256,
                     "The size of video cluster.")
flags.DEFINE_integer("audio_cluster_size", 32,
                     "The size of audio cluster.")
flags.DEFINE_integer("filter_size", 2,
                     "The filter multiplier size for deep context gate.")
flags.DEFINE_integer("hidden_size", 1024,
                     "The number of units after attention cluster layer.")
flags.DEFINE_bool("shift_operation", True,
                  "True iff shift operation is on.")
flags.DEFINE_float("cluster_dropout", 0.7,
                   "Dropout rate for clustering operation")
flags.DEFINE_float("ff_dropout", 0.8,
                   "Dropout rate for Feed Forward operation")

FLAGS = flags.FLAGS


class AttentionClusterModule(modules.BaseModule):
    def __init__(self, feature_size, max_frames, dropout_rate, cluster_size,
                 add_batch_norm, shift_operation, is_training):
        """ Initialize AttentionClusterModule.
        :param feature_size: int
        :param max_frames: vector of int
        :param dropout_rate: float
        :param cluster_size: int
        :param add_batch_norm: bool
        :param shift_operation: bool
        :param is_training: bool
        """
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.dropout_rate = dropout_rate
        self.shift_operation = shift_operation
        self.cluster_size = cluster_size

    def forward(self, inputs, **unused_params):
        """ Forward method for AttentionClusterModule.
        :param inputs: 3D Tensor of size 'batch_size x max_frames x feature_size'
        :return: 2D Tensor of size 'batch_size x (feature_size * cluster_size)
        """
        inputs = tf.reshape(inputs, [-1, self.feature_size])
        reshaped_inputs = tf.reshape(inputs, [-1, self.max_frames, self.feature_size])

        attention_weights = tf.layers.dense(inputs, self.cluster_size, use_bias=False, activation=None)
        float_cpy = tf.cast(self.feature_size, dtype=tf.float32)
        attention_weights = tf.divide(attention_weights, tf.sqrt(float_cpy))
        if self.add_batch_norm:
            attention_weights = tf.layers.batch_normalization(attention_weights, training=self.is_training)
        if self.is_training:
            attention_weights = tf.nn.dropout(attention_weights, self.dropout_rate)
        attention_weights = tf.nn.softmax(attention_weights)

        reshaped_attention = tf.reshape(attention_weights, [-1, self.max_frames, self.cluster_size])
        transposed_attention = tf.transpose(reshaped_attention, perm=[0, 2, 1])
        # -> transposed_attention: batch_size x cluster_size x max_frames
        activation = tf.matmul(transposed_attention, reshaped_inputs)
        # -> activation: batch_size x cluster_size x feature_size
        transformed_activation = tf.transpose(activation, perm=[0, 2, 1])
        # -> transformed_activation: batch_size x feature_size x cluster_size
        transformed_activation = tf.nn.l2_normalize(transformed_activation, 1)

        if self.shift_operation:
            alpha = tf.get_variable("alpha",
                                    [self.cluster_size],
                                    initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable("beta",
                                   [self.cluster_size],
                                   initializer=tf.constant_initializer(0.0))
            transformed_activation = tf.multiply(transformed_activation, alpha)
            transformed_activation = tf.add(transformed_activation, beta)

        normalized_activation = tf.nn.l2_normalize(transformed_activation, 1)
        normalized_activation = tf.reshape(normalized_activation, [-1, self.cluster_size * self.feature_size])
        normalized_activation = tf.nn.l2_normalize(normalized_activation)

        return normalized_activation


class AttentionClusterModel(models.BaseModel):
    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     hidden_size=None,
                     is_training=True,
                     **unused_params):
        video_cluster_size = FLAGS.video_cluster_size
        audio_cluster_size = FLAGS.audio_cluster_size
        shift_operation = FLAGS.shift_operation
        cluster_dropout = FLAGS.cluster_dropout
        ff_dropout = FLAGS.ff_dropout_rate
        filter_size = FLAGS.filter_size
        hidden_size = FLAGS.hidden_size

        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        # Differentiate video & audio features.
        video_features = reshaped_input[:, 0:1024]
        audio_features = reshaped_input[:, 1024:]
        video_features = tf.nn.l2_normalize(video_features, 1)
        audio_features = tf.nn.l2_normalize(audio_features, 1)
        video_features = tf.reshape(video_features, [-1, max_frames, 1024])
        audio_features = tf.reshape(audio_features, [-1, max_frames, 128])

        video_cluster = AttentionClusterModule(feature_size=1024,
                                               max_frames=max_frames,
                                               dropout_rate=cluster_dropout,
                                               cluster_size=video_cluster_size,
                                               add_batch_norm=True,
                                               shift_operation=shift_operation,
                                               is_training=is_training)

        audio_cluster = AttentionClusterModule(feature_size=128,
                                               max_frames=max_frames,
                                               dropout_rate=cluster_dropout,
                                               cluster_size=audio_cluster_size,
                                               add_batch_norm=True,
                                               shift_operation=shift_operation,
                                               is_training=is_training)

        with tf.variable_scope("video"):
            video_cluster_activation = video_cluster.forward(video_features)

        with tf.variable_scope("audio"):
            audio_cluster_activation = audio_cluster.forward(audio_features)

        concat_activation = tf.concat([video_cluster_activation, audio_cluster_activation], 1)
        activation = tf.layers.dense(concat_activation, hidden_size, use_bias=False, activation=None)
        activation = tf.layers.batch_normalization(activation, training=is_training)

        # Deep context gating.
        gating_weights1 = tf.layers.dense(activation, hidden_size * filter_size,
                                          use_bias=False, activation=tf.nn.relu)
        gating_weights1 = tf.layers.batch_normalization(gating_weights1, training=is_training)
        if is_training:
            gating_weights1 = tf.nn.dropout(gating_weights1, ff_dropout)

        gating_weights2 = tf.layers.dense(gating_weights1, hidden_size, use_bias=False, activation=None)
        gating_weights2 = tf.layers.batch_normalization(gating_weights2, training=is_training)
        gating_weights2 = tf.sigmoid(gating_weights2)
        activation = tf.multiply(activation, gating_weights2)

        aggregated_model = getattr(video_level_models,
                                   "MoeModel")

        return aggregated_model().create_model(
            model_input=activation,
            filter_size=filter_size,
            vocab_size=vocab_size,
            is_training=is_training,
            ff_dropout=ff_dropout,
            **unused_params)
