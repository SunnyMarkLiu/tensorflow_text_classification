#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-10-10 下午7:21
"""
from __future__ import absolute_import, division, print_function

import os
import sys

import time
import math
import datetime

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

from fast_text_model import FastText
from utils import data_util
from conf.configure import Configure

# disable TF debug logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float('validate_split_percentage', 0.1, 'Percentage of the training data to use for validation')

# Model Hyperparameters
tf.flags.DEFINE_integer('embedding_dim', 300, 'Dimensionality of word embedding (default: 300)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, "Dropout keep probability (default: 0.5)")

# Training parameters
tf.flags.DEFINE_integer("max_learning_rate", 0.001, "Max learning_rate when start training (default: 0.01)")
tf.flags.DEFINE_integer("min_learning_rate", 0.00001, "Min learning_rate when start training (default: 0.0001)")
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("train_verbose_every_steps", 50, "Show the training info every steps (default: 100)")
tf.flags.DEFINE_integer("evaluate_every_steps", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every_steps", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("max_num_checkpoints_to_keep", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("decay_coefficient", 5.0, "Decay coefficient (default: 2.5)")

FLAGS = tf.flags.FLAGS

print(' ================ Training Parameters ================')
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))

embedding_dimension = FLAGS.embedding_dim

# Data Preparation
# ==================================================
print('---> load train text dataset')
# datasets = data_util.get_datasets_20newsgroup()
# x_text, y = data_util.load_data_labels(datasets)
x_text, y = data_util.load_text_datasets()

print('---> build vocabulary according this text dataset')
max_document_length = max([len(x.split(" ")) for x in x_text])
print('max_document_length = {}'.format(max_document_length))
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

print('---> Split train/validate set')
valid_sample_index = -1 * int(FLAGS.validate_split_percentage * float(len(y)))
x_train, x_valid = x[:valid_sample_index], x[valid_sample_index:]
y_train, y_valid = y[:valid_sample_index], y[valid_sample_index:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_valid)))
print('---> create train/valid data wapper')
train_data_wrapper = data_util.DataWrapper(x_train, y_train, istrain=True)
valid_data_wrapper = data_util.DataWrapper(x_valid, y_valid, istrain=False)

print('---> build model')
# Built model and start training
# ==================================================
with tf.Graph().as_default(), tf.device('/gpu:2'):
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    session = tf.Session(config=session_conf)
    with session.as_default():
        fast_text = FastText(embedding_dim=embedding_dimension,
                             sequence_length=max_document_length,
                             label_size=2,
                             vocabulary_size=vocabulary_size,
                             embedding_trainable=False)

        # Define global training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=fast_text.learning_rate)
        # train_op = optimizer.minimize(fast_text.loss, global_step=global_step)
        grads_and_vars = optimizer.compute_gradients(
            fast_text.loss)  # Compute gradients of `loss` for the variables in `var_list`.
        # some other operation to grads_and_vars, eg. cap the gradient
        # ...
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)  # Apply gradients to variables.

        # Keep track of gradient values and sparsity distribution (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(name="{}/grad/hist".format(v.name), values=g)
                # The fraction of zeros in gradient
                grad_sparsity_summary = tf.summary.scalar("{}/grad/parsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(grad_sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(time.time())
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "run", timestamp))
        print("---> write logs and summaries to {}".format(out_dir))

        # Summaries for loss and accuracy
        learning_rate_summary = tf.summary.scalar("learning_rate", fast_text.learning_rate)
        loss_summary = tf.summary.scalar("loss", fast_text.loss)
        accuracy_summary = tf.summary.scalar("accuracy", fast_text.accuracy)

        # Merge Summaries for train
        train_summary_op = tf.summary.merge([learning_rate_summary, loss_summary, accuracy_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(logdir=train_summary_dir, graph=session.graph)

        # Merge Summaries for valid
        valid_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        valid_summary_dir = os.path.join(out_dir, "summaries", "valid")
        valid_summary_writer = tf.summary.FileWriter(logdir=valid_summary_dir, graph=session.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.join(out_dir, "checkpoints")
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_num_checkpoints_to_keep)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocabulary"))

        # Initialize all variables
        session.run(tf.global_variables_initializer())

        print('---> start training...')
        vocabulary = vocab_processor.vocabulary_
        print('---> load pre-trained word vectors')
        word_embeddings = data_util.load_glove_word_embeddings(vocabulary,
                                                               Configure.glove_word_embedding_file,
                                                               vector_size=300)
        print('---> assign model embedding matrix')
        session.run(tf.assign(ref=fast_text.word_embedding_matrix, value=word_embeddings))


        def train_step(x_batch, y_batch, learning_rate_):
            """ training step """
            feed_dict = {
                fast_text.sentence: x_batch,
                fast_text.labels: y_batch,
                fast_text.learning_rate: learning_rate_
            }
            _, step, summaries, loss, accuracy = session.run([train_op, global_step, train_summary_op,
                                                              fast_text.loss, fast_text.accuracy],
                                                             feed_dict=feed_dict)
            if step % FLAGS.train_verbose_every_steps == 0:
                time_str = datetime.datetime.now().isoformat()
                print("train {}: step {}, loss {:g}, acc {:g}, learning_rate {:g}"
                      .format(time_str, step, loss, accuracy, learning_rate_))

            train_summary_writer.add_summary(summaries, step)


        def valid_step(x_batch, y_batch, valid_writer=None):
            """ validate step """
            feed_dict = {
                fast_text.sentence: x_batch,
                fast_text.labels: y_batch
            }
            step, summaries, loss, accuracy = session.run([global_step, valid_summary_op,
                                                           fast_text.loss, fast_text.accuracy],
                                                          feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("---> valid {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if valid_writer:
                valid_writer.add_summary(summaries, current_step)


        max_learning_rate = FLAGS.max_learning_rate
        min_learning_rate = FLAGS.min_learning_rate

        total_train_steps = FLAGS.epochs * (x_train.shape[0] // FLAGS.batch_size)
        decay_speed = FLAGS.decay_coefficient * len(y_train) / FLAGS.batch_size

        print('---> total train steps: {}'.format(total_train_steps))
        counter = 0
        for epoch in range(FLAGS.epochs):
            print('-----> train epoch: {:d}/{:d}'.format(epoch + 1, FLAGS.epochs))
            for _ in range(x_train.shape[0] // FLAGS.batch_size):
                learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-counter / decay_speed)
                counter += 1

                batch_x, batch_y = train_data_wrapper.next_batch(FLAGS.batch_size)

                train_step(batch_x, batch_y, learning_rate)
                current_step = tf.train.global_step(session, global_step)

                if current_step % FLAGS.evaluate_every_steps == 0:
                    print('---> perform validate')
                    valid_step(batch_x, batch_y, valid_summary_writer)

                if current_step % FLAGS.checkpoint_every_steps == 0:
                    path = saver.save(session, checkpoint_prefix, global_step=global_step)
                    print('---> save model to checkpoint: {}'.format(path))
