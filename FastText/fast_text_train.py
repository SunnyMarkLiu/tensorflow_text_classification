#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-10-10 下午7:21
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

from fast_text_model import FastText
from utils import data_util

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float('validate_split_percentage', 0.1, 'Percentage of the training data to use for validation')

# Model Hyperparameters
tf.flags.DEFINE_integer('embedding_dim', 300, 'Dimensionality of word embedding (default: 300)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, "Dropout keep probability (default: 0.5)")

# Training parameters
tf.flags.DEFINE_integer("max_learning_rate", 0.01, "Max learning_rate when start training (default: 0.01)")
tf.flags.DEFINE_integer("min_learning_rate", 0.0001, "Min learning_rate when start training (default: 0.0001)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every_steps", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every_steps", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints_stored", 5, "Number of checkpoints to store (default: 5)")

FLAGS = tf.flags.FLAGS

print(' ================ Training Parameters ================')
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))

embedding_dimension = FLAGS.embedding_dim

# Data Preparation
# ==================================================
print('---> load text dataset')
x_text, y = data_util.load_text_datasets()

print('---> build vocabulary according this text dataset')
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_document_length)
# Maps documents to sequences of word ids in vocabulary
x = np.array(list(vocab_processor.fit_transform(x_text)))

vocabulary_size = len(vocab_processor.vocabulary_)
print('built vocabulary size: {:d}'.format(vocabulary_size))
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x = x[shuffle_indices]
y = y[shuffle_indices]

# Split train/validate set
dev_sample_index = -1 * int(FLAGS.validate_split_percentage * float(len(y)))
x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    session = tf.Session(session_conf)
    with session.as_default():
        fast_text = FastText(embedding_dim=embedding_dimension,
                             sequence_length=max_document_length,
                             label_size=2,
                             vocabulary_size=vocabulary_size,
                             embedding_trainable=False)


