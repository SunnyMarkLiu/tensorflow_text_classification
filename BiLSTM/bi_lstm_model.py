#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-10-30 下午8:22
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import tensorflow as tf
from tensorflow.contrib import rnn


class BiLSTM(object):
    def __init__(self, sequence_length, label_size, vocabulary_size,
                 embedding_dim, hidden_size,
                 embedding_trainable=False, l2_reg_lambda=0.0,
                 lstm_drop_out=False, input_keep_prob=1.0,
                 output_keep_prob=1.0, state_keep_prob=1.0):
        self.sequence_length = sequence_length
        self.label_size = label_size
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.embedding_trainable = embedding_trainable
        self.hidden_size = hidden_size,
        self.l2_reg_lambda = l2_reg_lambda
        self.lstm_drop_out = lstm_drop_out
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob
        self.state_keep_prob = state_keep_prob

        # Placeholders for input, output
        self.sentence = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name='input_x')
        self.labels = tf.placeholder(tf.int8, shape=[None, self.label_size], name='input_y')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # build bilstm model architecture
        self.build_model()


    def build_model(self):
        """
        build bilstm model architecture
        1. embeddding layer, 2.Bi-LSTM layer, 3.concat, 4.FC layer 5.softmax
        """
        # 1. Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.embedding_matrix = tf.get_variable(shape=[self.vocabulary_size, self.embedding_dim],
                                                    initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                                    name='embedding_matrix',
                                                    trainable=self.embedding_trainable)

            # get emebedding of words in the sentence, [None, sequence_length, embedding_dim]
            self.embedded_sentence = tf.nn.embedding_lookup(self.embedding_matrix, self.sentence)

        # 2. Bi-LSTM layer
        with tf.name_scope('bilstm_layer'):
            fw_lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_size)  # forward direction cell
            bw_lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_size)  # backward direction cell

            if self.lstm_drop_out:
                fw_lstm_cell = rnn.DropoutWrapper(cell=fw_lstm_cell,
                                                  input_keep_prob=self.input_keep_prob,
                                                  output_keep_prob=self.output_keep_prob,
                                                  state_keep_prob=self.state_keep_prob)
                bw_lstm_cell = rnn.DropoutWrapper(cell=bw_lstm_cell,
                                                  input_keep_prob=self.input_keep_prob,
                                                  output_keep_prob=self.output_keep_prob,
                                                  state_keep_prob=self.state_keep_prob)

            '''
            bidirectional_dynamic_rnn: input:  [batch_size, sequence_length, embedding_dim], max_time == sequence_length
                                       output: A tuple (outputs, output_states)
                                            outputs: A tuple (output_fw, output_bw)
                                               output_fw: [batch_size, max_time, cell_fw.output_size]
                                               output_bw: [batch_size, max_time, cell_bw.output_size]
            '''
            (fw_output, bw_output), _ = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell, bw_lstm_cell, self.embedded_sentence, dtype=tf.float32)
            print("bidirectional_dynamic_rnn outputs: ", fw_output, bw_output)

            # 3. concat, axis=2, concat cell_fw.output_size and cell_bw.output_size
            output_rnn = tf.concat((fw_output, bw_output), axis=2) # [batch_size, sequence_length, hidden_size * 2]
            print("concate output_rnn: ", output_rnn)

            self.output_rnn_last = tf.reduce_mean(output_rnn, axis=1) # [batch_size, hidden_size * 2]
            print("output_rnn_last: ", self.output_rnn_last)

        with tf.name_scope('readout'):
            # 4.linear classifier
            self.W_projection = tf.get_variable(shape=[self.hidden_size * 2, self.label_size],
                                                initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                                name='linear_W_projection')
            self.b_projection = tf.get_variable(shape=[self.label_size],
                                                name='linear_b_projection')

            self.logits = tf.add(tf.matmul(self.output_rnn_last, self.W_projection), self.b_projection, name='logits')

        with tf.name_scope("loss"):
            l2_loss = tf.constant(0)
            if self.embedding_trainable:
                l2_loss += tf.nn.l2_loss(self.embedding_matrix)

            # l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            l2_loss += tf.nn.l2_loss(self.W_projection)
            l2_loss += tf.nn.l2_loss(self.b_projection)

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            labels = tf.argmax(self.labels, 1)
            self.predictions = tf.argmax(self.logits, 1, name="predictions")
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, labels), "float"), name="accuracy")
