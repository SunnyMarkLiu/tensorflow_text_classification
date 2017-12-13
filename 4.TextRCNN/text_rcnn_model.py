#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-12-8 下午3:21
"""
from __future__ import absolute_import, division, print_function

import copy
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import tensorflow as tf


class TextRCNN(object):
    def __init__(self, label_size, sequence_length, vocabulary_size, embedding_dim, batch_size,
                 latent_hidden_size, embedding_trainable=False, l2_reg_lambda=0.0,
                 initializer=tf.random_normal_initializer(stddev=0.1)
                 ):
        # set hyperparamter
        self.label_size = label_size
        self.sequence_length = sequence_length
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.embedding_trainable = embedding_trainable

        # transform to latent semantic vector
        self.latent_hidden_size = latent_hidden_size

        self.batch_size = batch_size
        self.activation = tf.nn.tanh

        self.l2_reg_lambda = l2_reg_lambda
        self.initializer = initializer

        # Placeholders for input, output
        self.sentence = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name='input_x')
        self.labels = tf.placeholder(tf.int8, shape=[None, self.label_size], name='input_y')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # instantiate weights
        self.instantiate_weights()

        # build bilstm model architecture
        self.build_model()

    def instantiate_weights(self):
        """
        define all model weights
        """
        with tf.name_scope("weights"):
            self.embedding_matrix = tf.get_variable(shape=[self.vocabulary_size, self.embedding_dim],
                                                    initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                                    name='embedding_matrix',
                                                    trainable=self.embedding_trainable)

            # according to paper
            self.left_side_first_word = tf.get_variable("left_side_first_word",
                                                        shape=[self.batch_size, self.embedding_dim],
                                                        initializer=self.initializer)
            self.right_side_last_word = tf.get_variable("right_side_last_word",
                                                        shape=[self.batch_size, self.embedding_dim],
                                                        initializer=self.initializer)
            # W (l) is a matrix that transforms the hidden layer (context) into the next hidden layer.
            self.W_l =  tf.get_variable("W_l",shape=[self.embedding_dim, self.embedding_dim],initializer=self.initializer)
            # W (sl) is a matrix that is used to combine the semantic of the current word with the next word’s left context
            self.W_sl = tf.get_variable("W_sl", shape=[self.embedding_dim, self.embedding_dim], initializer=self.initializer)
            self.W_r = tf.get_variable("W_r", shape=[self.embedding_dim, self.embedding_dim], initializer=self.initializer)
            self.W_sr = tf.get_variable("W_sr", shape=[self.embedding_dim, self.embedding_dim], initializer=self.initializer)

            # y(i_2) layer weights, generate latent semantic vector
            self.latent_W = tf.get_variable("latent_W",shape=[self.embedding_dim * 3, self.latent_hidden_size],initializer=self.initializer)
            self.latent_b = tf.get_variable("latent_b", shape=[self.latent_hidden_size])

            # fc layer
            self.fc_W = tf.get_variable("fc_W",shape=[self.latent_hidden_size, self.label_size],initializer=self.initializer)
            self.fc_b = tf.get_variable("fc_b", shape=[self.label_size])


    def build_model(self):
        """
        build text rcnn model architecture
        1. embeddding layer, 2.Conv_recurrent layer, 3.max-pooling, 4.FC layer 5.softmax
        """
        # 1. Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            # get emebedding of words in the sentence, [None, sequence_length, embedding_dim]
            self.embedded_words = tf.nn.embedding_lookup(self.embedding_matrix, self.sentence)

        # 2. Convolutional layer with recurrent structure
        with tf.name_scope('conv_recurrent_layer'):
            # [None, sentence_length, latent_hidden_size]
            conv_recurrent_out = self.conv_layer_with_recurrent_structure()

        # 3. max-pooling
        with tf.name_scope('max_pooling_layer'):
            max_pooling_out = tf.reduce_max(conv_recurrent_out, axis=1)

        print('---> max_pooling_out', max_pooling_out)
        # 4. FC layer
        with tf.name_scope('readout'):
            h_dropout = tf.nn.dropout(max_pooling_out, keep_prob=self.dropout_keep_prob)
            self.logits = tf.add(tf.matmul(h_dropout, self.fc_W), self.fc_b, name='logits')

        with tf.name_scope("loss"):
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_losses

        with tf.name_scope("accuracy"):
            labels = tf.argmax(self.labels, 1)
            self.predictions = tf.argmax(self.logits, 1, name="predictions")
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, labels), "float"), name="accuracy")

    def conv_layer_with_recurrent_structure(self):
        """
        Convolutional layer with recurrent structure
        input: self.embedded_words [None, sentence_length, embedding_dim]
        :return: [None, sentence_length, embedding_dim * 3]
        """
        # 1. get splitted list of word embeddings
        # sentence_length [None,1,embed_size]
        embedded_words_split = tf.split(self.embedded_words, self.sequence_length, axis=1)
        # sentence_length [None,embed_size]
        embedded_words_squeezed = [tf.squeeze(x, axis=1) for x in embedded_words_split]

        embedding_previous = self.left_side_first_word
        context_left_previous = tf.zeros((self.batch_size, self.embedding_dim))

        # 2. get list of context left
        context_left_list = []
        for i, current_embedding_word in enumerate(embedded_words_squeezed):
            context_left = self.get_context_left(context_left_previous, embedding_previous)  # [None, embed_size]
            context_left_list.append(context_left)

            # next word
            embedding_previous = current_embedding_word
            context_left_previous = context_left

        # 3. get context right
        embedded_words_squeezed2 = copy.copy(embedded_words_squeezed)
        embedded_words_squeezed2.reverse()

        embedding_afterward = self.right_side_last_word
        context_right_afterward = tf.zeros((self.batch_size, self.embedding_dim))
        context_right_list = []
        for j, current_embedding_word in enumerate(embedded_words_squeezed2):
            context_right = self.get_context_right(context_right_afterward, embedding_afterward)
            context_right_list.append(context_right)
            context_right_afterward = context_right
            embedding_afterward = current_embedding_word

        # 4.ensemble left,embedding,right to output
        output_representations_list = [] # sentence_length [None, embedding_dim]
        for index, current_embedding_word in enumerate(embedded_words_squeezed):
            # [None, embedding_dim]
            word_representation = tf.concat([context_left_list[index],
                                            current_embedding_word,
                                            context_right_list[index]], axis=1)

            # trainform latent semantic vector
            word_representation = self.activation(tf.add(tf.matmul(word_representation, self.latent_W), self.latent_b))
            output_representations_list.append(word_representation)

        # 5. stack list to a tensor
        # [None, sentence_length, latent_hidden_size]
        conv_recurrent_output = tf.stack(output_representations_list, axis=1)
        return conv_recurrent_output


    def get_context_left(self, context_left, embedding_previous):
        """
        :param context_left:
        :param embedding_previous:
        :return: output:[None,embed_size]
        """
        # context_left:[batch_size, embed_size]; W_l:[embed_size, embed_size]
        left_c = tf.matmul(context_left, self.W_l)
        # embedding_previous: [batch_size,embed_size]
        left_e = tf.matmul(embedding_previous, self.W_sl)
        left_h = left_c + left_e
        context_left = self.activation(left_h)
        return context_left

    def get_context_right(self, context_right, embedding_afterward):
        """
        :param context_right:
        :param embedding_afterward:
        :return: output:[None,embed_size]
        """
        right_c = tf.matmul(context_right, self.W_r)
        right_e = tf.matmul(embedding_afterward, self.W_sr)
        right_h = right_c + right_e
        context_right = self.activation(right_h)
        return context_right
