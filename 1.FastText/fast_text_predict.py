#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-10-12 下午1:36
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

from utils import data_util

# disable TF debug logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

FLAGS = tf.flags.FLAGS

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
# ==================================================

print('---> load test text dataset')
# datasets = data_util.get_datasets_20newsgroup()
# x_text, y = data_util.load_data_labels(datasets)
x_text, y = data_util.load_text_datasets()

# Restore vocabulary in checkpoint dir
vocabulary_path = os.path.join(FLAGS.checkpoint_dir, '..', 'vocabulary')
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(filename=vocabulary_path)

# map text to vocabulary index
x_text = np.array(list(vocab_processor.transform(x_text)))
print('---> create test data wapper')
test_data_wrapper = data_util.DataWrapper(x_text, istrain=False, is_shuffle=False)

# Evaluation
# ==================================================

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print(checkpoint_file)
with tf.Graph().as_default(), tf.device('/gpu:2'):
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    session = tf.Session(config=session_conf)
    with session.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(session, checkpoint_file)

        # Get the placeholders from the graph by name for predict
        input_x = session.graph.get_operation_by_name('input_x').outputs[0]

        # feedforward read out probabilities
        logits = session.graph.get_operation_by_name('readout/logits').outputs[0]

        print('---> predict test data')
        all_probabilities = None
        for _ in range(test_data_wrapper.x.shape[0] // FLAGS.batch_size + 1):
            test_x, __ = test_data_wrapper.next_batch(FLAGS.batch_size)
            batch_logits = session.run(logits, feed_dict={input_x: test_x})
            batch_probabilities = softmax(batch_logits)
            if all_probabilities is not None:
                all_probabilities = np.concatenate([all_probabilities, batch_probabilities])
            else:
                all_probabilities = batch_probabilities

        print('all_probabilities:{}'.format(all_probabilities.shape))
