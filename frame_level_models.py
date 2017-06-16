# Copyright 2016 Google Inc. All Rights Reserved.
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

"""Contains a collection of models which operate on variable-length sequences.
"""
import math
import sys

import models
import video_level_models
import tensorflow as tf
import model_utils as utils
import weights_rwa

import tensorflow.contrib.slim as slim
from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")

class FrameLevelLogisticModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the
    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators

    output = slim.fully_connected(
        avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}

class DbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    #tf.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    cluster_weights = tf.get_variable("cluster_weights",
      [feature_size, cluster_size],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
    #tf.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.matmul(reshaped_input, cluster_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="cluster_bn")
    else:
      cluster_biases = tf.get_variable("cluster_biases",
        [cluster_size],
        initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
      #tf.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    activation = tf.nn.relu6(activation)
    #tf.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])
    activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

    hidden1_weights = tf.get_variable("hidden1_weights",
      [cluster_size, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    #tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(activation, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      #tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = tf.nn.relu6(activation)
    #tf.summary.histogram("hidden1_output", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        **unused_params)

class LstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ], state_is_tuple=False)

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class BNGRUModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    stacked_rnn = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.GRUCell(lstm_size)
                for _ in range(number_of_layers)
                ], state_is_tuple=False)

    outputs, state = tf.nn.dynamic_rnn(stacked_rnn, model_input,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    state = slim.batch_norm(
        state,
        center=True,
        scale=True,
        is_training=True,
        scope='proj')

    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)



class GruModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.GRUCell(lstm_size)
                for _ in range(number_of_layers)
                ], state_is_tuple=False)

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)


class BiGRUModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    with tf.variable_scope('fw'):
        rnn_fw = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.GRUCell(lstm_size)
                for _ in range(number_of_layers)
                ], state_is_tuple=False)


    with tf.variable_scope('bw'):
        rnn_bw = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.GRUCell(lstm_size)
                for _ in range(number_of_layers)
                ], state_is_tuple=False)

    outputs, state = tf.nn.bidirectional_dynamic_rnn(rnn_fw, rnn_bw, model_input,
                                       sequence_length=num_frames,
                                       dtype=tf.float32, swap_memory=True)
    state = tf.concat(state, axis=1)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    state = slim.batch_norm(
          state,
          center=True,
          scale=True,
          is_training=True,
          scope='proj')

    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

"""
Copyright (c) 2017, University of Texas Southwestern Medical Center
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the University of Texas at Austin nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Recurrent Weighted Average

Implementation modified from: https://github.com/jostmey/rwa

Paper:
@article{ostmeyer2017machine,
  title={Machine Learning on Sequential Data Using a Recurrent Weighted Average},
  author={Ostmeyer, Jared and Cowell, Lindsay},
  journal={arXiv preprint arXiv:1703.01253},
  year={2017}
}

"""

class RwaModel(models.BaseModel):


    def create_model(self, model_input, vocab_size, num_frames, **unused_params):

        # constants

        init_factor = 1.0
        num_cells = FLAGS.lstm_cells
        input_shape = model_input.get_shape().as_list()
        batch_size, max_steps, num_features = input_shape

        # trainable weights
        s = weights_rwa.init_state(num_cells, "s", init_factor)
        W_g = weights_rwa.init_weight([num_features+num_cells, num_cells], "W_g")
        W_u = weights_rwa.init_weight([num_features, num_cells], "W_u")
        W_a = weights_rwa.init_weight([num_features+num_cells, num_cells], "W_a")
        b_g = weights_rwa.init_bias(num_cells, "b_g")
        b_u = weights_rwa.init_bias(num_cells, "b_u")
        b_a = weights_rwa.init_bias(num_cells, "b_a")

        #pl = tf.placeholder(tf.float32, shape=[None, num_cells])
        pl = tf.reshape(model_input, [-1, max_steps*num_features])[:, :num_cells]

        # internal states
        #n = tf.zeros([batch_size, num_cells])
        #d = tf.zeros([batch_size, num_cells])
        #h = tf.zeros([batch_size, num_cells])
        #a_max = tf.fill([batch_size, num_cells], -1E38) # Start off with lowest number possible
        n = tf.zeros_like(pl)
        d = tf.zeros_like(pl)
        h = tf.zeros_like(pl)
        a_max = tf.multiply(tf.ones_like(pl), -1E38)

        # define model
        h += tf.nn.tanh(tf.expand_dims(s, 0))

        for i in range(max_steps):

            x_step = model_input[:,i,:]
            xh_join = tf.concat(axis=1, values=[x_step, h]) # Combine the features and hidden state into one tensor

            u = tf.matmul(x_step, W_u)+b_u
            g = tf.matmul(xh_join, W_g)+b_g
            a = tf.matmul(xh_join, W_a)     # The bias term when factored out of the numerator and denominator cancels and is unnecessary

            z = tf.multiply(u, tf.nn.tanh(g))

            a_newmax = tf.maximum(a_max, a)
            exp_diff = tf.exp(a_max-a_newmax)
            exp_scaled = tf.exp(a-a_newmax)

            n = tf.multiply(n, exp_diff)+tf.multiply(z, exp_scaled) # Numerically stable update of numerator
            d = tf.multiply(d, exp_diff)+exp_scaled # Numerically stable update of denominator
            h_new = tf.nn.tanh(tf.div(n, d))
            a_max = a_newmax

            h = tf.where(tf.greater(num_frames, i), h_new, h)    # Use new hidden state only if the sequence length has not been exceeded


        aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=h,
            vocab_size=vocab_size,
            **unused_params)



class DropoutGruModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.GRUCell(lstm_size), 0.9, 0.9)
                for _ in range(number_of_layers)
                ], state_is_tuple=False)

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)




class ResRnnModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    #from rnn_cell_modern import Delta_RNN as drnn
    from rnn_wrappers_modern import MultiRNNCell as mrnn

    cells = []
    for i in range(number_of_layers):
        with tf.variable_scope('cell_'+str(i)):
            cells.append(tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0))

    stacked_rnn = mrnn(cells, use_residual_connections=True, state_is_tuple=True)

    outputs, state = tf.nn.dynamic_rnn(stacked_rnn, model_input,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)


class LateVladModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    model_input = utils.SampleRandomSequence(model_input, num_frames, 128)

    input_v = model_input[:,:,:1024]
    input_a = model_input[:,:,1024:]

    K = 8

    with tf.variable_scope('video'):
        x = input_v
        input_shape = x.get_shape().as_list()
        _, N, D = input_shape
        c_bound = math.sqrt(1. / (K * D))
        c = tf.get_variable(name='c',
                            shape=[K, N],
                            dtype=tf.float32,
                            initializer=tf.random_uniform_initializer(-c_bound, c_bound))
        a = slim.convolution(x,
                             num_outputs=K,
                             kernel_size=1,
                             data_format='NWC',
                             scope='conv')
        a = tf.nn.softmax(a)
        v = []
        for k in range(K):
          t = x-c[k][None, :, None]
          t = tf.multiply(t, a[:,:,k][:,:,None])
          t = tf.reduce_sum(t, 1)
          t = tf.nn.l2_normalize(t, dim=1)
          v.append(t)
        v = tf.stack(v, axis=1)
        v = tf.reshape(v, [-1, K*D])

        proj_weights = tf.get_variable("proj_weights",
          [K*D, 1024],
          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(K*D)))
        activation_v = tf.matmul(v, proj_weights)

    with tf.variable_scope('audio'):
        x = input_a
        input_shape = x.get_shape().as_list()
        _, N, D = input_shape
        c_bound = math.sqrt(1. / (K * D))
        c = tf.get_variable(name='c',
                            shape=[K, N],
                            dtype=tf.float32,
                            initializer=tf.random_uniform_initializer(-c_bound, c_bound))
        a = slim.convolution(x,
                             num_outputs=K,
                             kernel_size=1,
                             data_format='NWC',
                             scope='conv')
        a = tf.nn.softmax(a)
        v = []
        for k in range(K):
          t = x-c[k][None, :, None]
          t = tf.multiply(t, a[:,:,k][:,:,None])
          t = tf.reduce_sum(t, 1)
          t = tf.nn.l2_normalize(t, dim=1)
          v.append(t)
        v = tf.stack(v, axis=1)
        v = tf.reshape(v, [-1, K*D])

        proj_weights = tf.get_variable("proj_weights",
          [K*D, 1024],
          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(K*D)))
        activation_a = tf.matmul(v, proj_weights)

    activation = tf.concat([activation_v, activation_a], axis=1)

    activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=True,
          scope='proj')

    activation = tf.nn.relu6(activation)


    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        **unused_params)


