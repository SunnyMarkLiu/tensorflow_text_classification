#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-10-16 上午11:17
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import tensorflow as tf


class TextCNN(object):
    def __init__(self, embedding_dim, sequence_length, label_size, vocabulary_size,
                 filter_sizes, num_filters, embedding_trainable=False, l2_reg_lambda=0.0):
        """
        init the model with hyper-parameters etc
        @:param filter_sizes: list
        """
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.label_size = label_size
        self.vocabulary_size = vocabulary_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.embedding_trainable = embedding_trainable

        # Placeholders for input, output
        self.sentence = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.labels = tf.placeholder(tf.float32, [None, self.label_size], name="input_y")
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.dropout_keep_ratio = tf.placeholder(tf.float32, name='dropout_keep_ratio')

        self.l2_reg_lambda = l2_reg_lambda

        # built fasttext model architecture
        self.built_model()

    def built_model(self):
        """
        built text cnn model architecture
        """
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.word_embedding_matrix = tf.get_variable(shape=[self.vocabulary_size, self.embedding_dim],
                                                         initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                                         name="word_embedding_matrix",
                                                         trainable=self.embedding_trainable)
            # 1.bag of words embedding
            inputwords_embeddings = tf.nn.embedding_lookup(self.word_embedding_matrix,
                                                           self.sentence)  # [None, self.sentence_len, self.embedding_dim]
            # 2.add feature map dimension, so axis=-1, like image channel
            self.inputwords_embeddings = tf.expand_dims(inputwords_embeddings,
                                                        axis=-1)  # [None, self.sentence_len, self.embedding_dim, 1]

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv_maxpool_{}'.format(filter_size)):
                # Convolution layer
                # [filter_height, filter_width, in_channels, out_channels]
                filter_shape = [filter_size, self.embedding_dim, 1, self.num_filters]
                W = tf.Variable(initial_value=tf.truncated_normal(filter_shape, mean=0.0, stddev=0.1), name='conv_w')
                b = tf.Variable(initial_value=tf.constant(0.1, shape=[self.num_filters]), name='conv_b')
                # strides=[1，stride，stride，1]
                conv = tf.nn.relu(
                    tf.nn.bias_add(
                        tf.nn.conv2d(self.inputwords_embeddings, W, strides=[1, 1, 1, 1], padding="VALID", name='conv'),
                        b
                    )
                )  # [batch, sequence_length - filter_size + 1, 1, filters]
                # Maxpooling
                max_pooled = tf.nn.max_pool(conv,
                                            ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1],  # strides=[1，stride，stride，1]
                                            padding='VALID',
                                            name='pool')  # [batch, 1, 1, filters]
                pooled_outputs.append(max_pooled)

        # Combine all the pooled features
        total_num_filters = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, axis=3)  # concate channels
        self.h_pool_flat = tf.reshape(h_pool, shape=[-1, total_num_filters])  # [-1, total_num_filters]

        # dropout
        with tf.name_scope('dropout_layer'):
            self.dropout = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_ratio)

        # full connected layer readout
        with tf.name_scope('readout_logits'):
            self.fc_W = tf.get_variable(name='fc_W',
                                        shape=[total_num_filters, self.label_size],
                                        initializer=tf.contrib.layers.xavier_initializer(uniform=True))
            self.fc_b = tf.Variable(initial_value=tf.constant(0.1, shape=[self.label_size]), name='fc_b')
            self.logits = tf.add(tf.matmul(self.dropout, self.fc_W), self.fc_b, name='logits')
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        with tf.name_scope('loss'):
            l2_loss = tf.constant(0.0)
            if self.embedding_trainable:
                l2_loss += tf.nn.l2_loss(self.word_embedding_matrix)

            l2_loss += tf.nn.l2_loss(self.fc_W)
            l2_loss += tf.nn.l2_loss(self.fc_b)

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        with tf.name_scope('accuracy'):
            true_labels = tf.argmax(self.labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, true_labels), "float"), name="accuracy")
