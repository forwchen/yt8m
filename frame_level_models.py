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

import models
import video_level_models
import tensorflow as tf
import model_utils as utils

import tensorflow.contrib.slim as slim
from tensorflow import flags
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("input_type", 3,
                     "input type.")
flags.DEFINE_string("video_level_classifier_model", "linear_res_mix_act_MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("conv_length", 3, "Receptive field of cnn.")
flags.DEFINE_integer("conv_hidden", 256, "Number of cnn hidden.")
flags.DEFINE_integer("conv_hidden1", 1024, "Number of cnn hidden.")
flags.DEFINE_integer("conv_hidden2", 1024, "Number of cnn hidden.")
flags.DEFINE_integer("conv_hidden3", 1024, "Number of cnn hidden.")
flags.DEFINE_integer("stride", 10, "Number of stride for short rnn.")

class audio_avgShort_twowayGRUModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
    """Creates a model which uses a Bidirectional GRU and mean audio features to represent the video.

                      ---->first half GRU----->
                      -                       -
    visual_feature ----                       concat---------------->
                      -                       -                     -
                      ---->second half GRU---->                     concat -----> video level classifier
                                                                    -
                                              mean audio features--->

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
    stride = FLAGS.stride
    max_frames = model_input.get_shape().as_list()[1]

    video_input = model_input[:,:,:1024]
    audio_input = model_input[:,:,1024:]

    first_num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    audio_den = tf.reshape(tf.tile(first_num_frames, [1, 128]), [-1, 128])
    mean_audio = tf.reduce_sum(audio_input, 1) / tf.maximum(audio_den, 1)

    pooled_input, num_frames = self.avg_pooled_func(video_input, num_frames, stride)

    pooled_input = slim.batch_norm(
      pooled_input,
      center=True,
      scale=True,
      is_training=is_training,
      scope="hidden1_bn")

    mean_audio = slim.batch_norm(
      mean_audio,
      center=True,
      scale=True,
      is_training=is_training,
      scope="hidden1_bn_audio")

    fw_gru = tf.contrib.rnn.GRUCell(lstm_size)
    bw_gru = tf.contrib.rnn.GRUCell(lstm_size)

    fw_outputs, fw_state = tf.nn.dynamic_rnn(fw_gru, pooled_input[:,:max_frames//(2*stride),:], 
        sequence_length=num_frames//2, dtype=tf.float32, scope='fw')
    bw_outputs, bw_state = tf.nn.dynamic_rnn(bw_gru, pooled_input[:,max_frames//(2*stride)::-1,:], 
        sequence_length=num_frames - num_frames//2, dtype=tf.float32, scope='bw')

    state = tf.concat([fw_state, bw_state], 1)
    state = tf.concat([state, mean_audio], 1)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

  def avg_pooled_func(self, model_input, num_frames_in, stride):
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    num_frames = num_frames_in // stride
    step = max_frames//stride

    first_layer_input = tf.reshape(model_input, [-1, stride, step, feature_size])
    first_layer_input = tf.reduce_sum(first_layer_input, 1)

    first_num_frames = tf.cast(tf.expand_dims(tf.expand_dims(num_frames, 1),2), tf.float32)
    denominators = tf.reshape(
        tf.tile(first_num_frames, [1, step, feature_size]), [-1, step, feature_size])
    first_layer_avg_pooled = first_layer_input / tf.maximum(denominators,1)

    return first_layer_avg_pooled, num_frames


class resav_ConvModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
    """Creates a model which uses a Convolutional model to represent the video.

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
    stride = FLAGS.stride
    conv_length = FLAGS.conv_length
    conv_hidden1 = FLAGS.conv_hidden1
    conv_hidden2 = FLAGS.conv_hidden2
    conv_hidden3 = FLAGS.conv_hidden3
    mean_feature = tf.reduce_mean(model_input, 1)
    feature_size = model_input.get_shape().as_list()[2]

    pooled_input = self.avg_pooled_func(model_input, stride)

    # To shape : 'batch_size' x 'max_frames' x 1 x 'num_features'
    input_expand = tf.expand_dims(pooled_input, -1)
    input_expand = tf.transpose(input_expand, [0,1,3,2])

    # conv_out : batch_size x max_frames-conv_length x 1 x conv_hidden
    conv_out = slim.conv2d(input_expand, conv_hidden1, [conv_length, 1], activation_fn=None, padding= 'SAME', scope='conv_1_1')
    conv_out = tf.nn.relu(slim.batch_norm(conv_out, center=True, scale=True, is_training=is_training, scope="bn_1_1"))
    conv_out = slim.conv2d(conv_out, conv_hidden1, [conv_length, 1], activation_fn=None, padding= 'SAME', scope='conv_1_2')
    conv_out = slim.batch_norm(conv_out, center=True, scale=True, is_training=is_training, scope="bn_1_2")
    res_out = slim.conv2d(input_expand, conv_hidden1, [conv_length, 1], activation_fn=None, padding= 'SAME', scope='xconv_1_1')
    res_out = res_out + conv_out
    res_out = slim.max_pool2d(res_out, [2,1], [2,1], scope='max_pool1')

    conv_out = slim.conv2d(res_out, conv_hidden2, [conv_length, 1], activation_fn=None, padding= 'SAME', scope='conv_2_1')
    conv_out = tf.nn.relu(slim.batch_norm(conv_out, center=True, scale=True, is_training=is_training, scope="bn_2_1"))
    conv_out = slim.conv2d(conv_out, conv_hidden2, [conv_length, 1], activation_fn=None, padding= 'SAME', scope='conv_2_2')
    conv_out = slim.batch_norm(conv_out, center=True, scale=True, is_training=is_training, scope="bn_2_2")
    res_out = slim.conv2d(res_out, conv_hidden2, [conv_length, 1], activation_fn=None, padding= 'SAME', scope='xconv_2_1')
    res_out = res_out + conv_out
    res_out = slim.max_pool2d(res_out, [2,1], [2,1], scope='max_pool2')

    conv_out = slim.conv2d(res_out, conv_hidden3, [conv_length, 1], activation_fn=None, padding= 'SAME', scope='conv_3_1')
    conv_out = tf.nn.relu(slim.batch_norm(conv_out, center=True, scale=True, is_training=is_training, scope="bn_3_1"))
    conv_out = slim.conv2d(conv_out, conv_hidden3, [conv_length, 1], activation_fn=None, padding= 'SAME', scope='conv_3_2')
    conv_out = slim.batch_norm(conv_out, center=True, scale=True, is_training=is_training, scope="bn_3_2")
    res_out = slim.conv2d(res_out, conv_hidden3, [conv_length, 1], activation_fn=None, padding= 'SAME', scope='xconv_3_1')
    res_out = res_out + conv_out
    res_out = slim.max_pool2d(res_out, [2,1], [2,1], scope='max_pool3')

    a = res_out.get_shape().as_list()[1]
    b = res_out.get_shape().as_list()[2]
    c = res_out.get_shape().as_list()[3]
    
    print(res_out.get_shape().as_list())
    
    res_out = tf.reshape(res_out, [-1, a*b*c])

    state = tf.concat([res_out, mean_feature], 1)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

  def avg_pooled_func(self, model_input, stride):
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    step = max_frames//stride

    first_layer_input = tf.reshape(model_input, [-1, stride, step, feature_size])
    first_layer_input = tf.reduce_mean(first_layer_input, 1)

    return first_layer_input

class pur_twowayGRUModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
    """Creates a model which uses a Bidirectional GRU without explictly using mean audio feature to represent the video.


                      ---->first half GRU----->
                      -                       -
    video_feature ----                       concat---------------->video level classifier
                      -                       -
                      ---->second half GRU---->


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
    stride = FLAGS.stride
    max_frames = model_input.get_shape().as_list()[1]

    pooled_input, num_frames = self.avg_pooled_func(model_input, num_frames, stride)

    pooled_input = slim.batch_norm(
      pooled_input,
      center=True,
      scale=True,
      is_training=is_training,
      scope="hidden1_bn")

  
    fw_gru = tf.contrib.rnn.GRUCell(lstm_size)
    bw_gru = tf.contrib.rnn.GRUCell(lstm_size)

    fw_outputs, fw_state = tf.nn.dynamic_rnn(fw_gru, pooled_input[:,:max_frames//(2*stride),:], 
        sequence_length=num_frames//2, dtype=tf.float32, scope='fw')
    bw_outputs, bw_state = tf.nn.dynamic_rnn(bw_gru, pooled_input[:,max_frames//(2*stride)::-1,:], 
        sequence_length=num_frames - num_frames//2, dtype=tf.float32, scope='bw')

    state = tf.concat([fw_state, bw_state], 1)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

  def avg_pooled_func(self, model_input, num_frames_in, stride):
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    num_frames = num_frames_in // stride
    step = max_frames//stride

    first_layer_input = tf.reshape(model_input, [-1, stride, step, feature_size])
    first_layer_input = tf.reduce_sum(first_layer_input, 1)

    first_num_frames = tf.cast(tf.expand_dims(tf.expand_dims(num_frames, 1),2), tf.float32)
    denominators = tf.reshape(
        tf.tile(first_num_frames, [1, step, feature_size]), [-1, step, feature_size])
    first_layer_avg_pooled = first_layer_input / tf.maximum(denominators,1)

    return first_layer_avg_pooled, num_frames

