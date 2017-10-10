#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-10-10 下午3:35
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import tensorflow as tf


class FastText(object):
    def __init__(self, embedding_dim, sequence_length, label_size, vocabulary_size, embedding_trainable=False):
        """
        init the model with hyper-parameters etc
        """
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.label_size = label_size
        self.vocabulary_size = vocabulary_size
        self.embedding_trainable = embedding_trainable

        # Placeholders for input, output and dropout
        self.sentence = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.labels = tf.placeholder(tf.float32, [None, self.label_size], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.learning_rate = tf.placeholder(tf.float32)

        # built fasttext model architecture
        self.built_model()

    def built_model(self):
        """
        The forward calculation from x to predict_y
        1.bag of words embedding --> 2.embeddings average -> 3.linear classifier
        :return: 
        """
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.word_embedding_matrix = tf.get_variable(shape=[self.vocabulary_size, self.embedding_dim],
                                                         name="word_embedding_matrix",
                                                         trainable=self.embedding_trainable)
            # 1.bag of words embedding
            sentence_embeddings = tf.nn.embedding_lookup(self.word_embedding_matrix,
                                                         self.sentence)  # [None, self.sentence_len, self.embedding_dim]
            # 2.embeddings average
            self.sentence_embedding = tf.reduce_mean(sentence_embeddings, axis=1)  # [None, self.embedding_dim]

        with tf.name_scope("readout"):
            # 3.linear classifier
            self.W = tf.get_variable(shape=[self.embedding_dim, self.label_size], name='W')
            self.b = tf.get_variable(shape=[self.label_size], name='b')
            self.logits = tf.add(tf.matmul(self.sentence_embedding, self.W), self.b, name='logits')

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            self.loss = tf.reduce_mean(losses)
