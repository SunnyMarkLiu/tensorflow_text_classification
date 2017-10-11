#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-10-9 下午8:28
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files

import cPickle
from conf.configure import Configure

import re
from string import punctuation
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer

word_tokenize = WordPunctTokenizer().tokenize
stop_words = ['the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 'which', 'this', 'that', 'these',
              'those', 'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while',
              'during', 'to', 'What', 'Which', 'Is', 'If', 'While', 'This']


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


class DataWrapper(object):
    def __init__(self, x, y=None, istrain=False):
        self.x = x
        self.y = y
        self.pointer = 0
        self.total_count = self.x.shape[0]
        self.istrain = istrain

    def shuffle(self):
        shuffled_index = np.arange(0, self.total_count)
        np.random.shuffle(shuffled_index)
        self.x = self.x[shuffled_index]
        if self.istrain:
            self.y = self.y[shuffled_index]

    def load_all_data(self):
        return self.next_batch(self.x.shape[0])

    def next_batch(self, batch_size):
        end = self.pointer + batch_size
        if end > self.total_count:
            end = self.total_count

        batch_x = self.x[self.pointer: end]
        batch_y = None
        if self.istrain:
            batch_y = self.y[self.pointer: end]

        self.pointer = end

        if self.pointer == self.total_count:
            self.shuffle()
            self.pointer = 0

        return batch_x, batch_y


def load_text_datasets():
    """
    Loads movie review dataset from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    :return: 
    """
    # Load data from files
    positive_examples = []
    negative_examples = []

    pos_files = os.listdir(Configure.positive_data_dir)
    for pf in pos_files:
        positive_examples.append(open(Configure.positive_data_dir + pf, "r").readline())

    neg_files = os.listdir(Configure.negative_data_dir)
    for nf in neg_files:
        negative_examples.append(open(Configure.negative_data_dir + nf, "r").readline())

    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [TextCleaner.clean_text(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def get_datasets_20newsgroup(subset='train', categories=None, shuffle=True, random_state=42):
    """
    Retrieve data from 20 newsgroups
    :param subset: train, test or all
    :param categories: List of newsgroup name
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the newsgroup
    """
    if categories is None:
        categories = ['alt.atheism', 'sci.space']
    datasets = fetch_20newsgroups(subset=subset, categories=categories, shuffle=shuffle, random_state=random_state)
    return datasets


def get_datasets_localdata(container_path=None, categories=None, load_content=True,
                           encoding='utf-8', shuffle=True, random_state=42):
    """
    Load text files with categories as subfolder names.
    Individual samples are assumed to be files stored a two levels folder structure.
    :param container_path: The path of the container
    :param categories: List of classes to choose, all classes are chosen by default (if empty or omitted)
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the dataset
    """
    datasets = load_files(container_path=container_path, categories=categories,
                          load_content=load_content, shuffle=shuffle, encoding=encoding,
                          random_state=random_state)
    return datasets


def load_data_labels(datasets):
    """
    Load data and labels
    :param datasets:
    :return:
    """
    # Split by words
    x_text = datasets['data']
    x_text = [TextCleaner.clean_text(sent) for sent in x_text]
    # Generate labels
    labels = []
    for i in range(len(x_text)):
        label = [0 for j in datasets['target_names']]
        label[datasets['target'][i]] = 1
        labels.append(label)
    y = np.array(labels)
    return [x_text, y]


def load_glove_word_embeddings(vocabulary, glove_w2c_filename, vector_size):
    """
    load embedding_vectors from the pre-trained glove vectors
    :param vocabulary: 
    :param glove_w2c_filename: 
    :param vector_size: 
    :return: [vocabulary_size, embedding_size]
    """
    # initial matrix with random uniform
    glove_word_embedding_matrix_path = Configure.glove_word_embedding_matrix
    if os.path.exists(glove_word_embedding_matrix_path):
        with open(glove_word_embedding_matrix_path, "rb") as f:
            return cPickle.load(f)

    word_embeddings = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(glove_w2c_filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            word_embeddings[idx] = vector
    f.close()

    with open(glove_word_embedding_matrix_path, "wb") as f:
        cPickle.dump(word_embeddings, f, -1)

    return word_embeddings


class TextCleaner(object):
    @staticmethod
    def clean_text(text, remove_stop_words=True, stem_words=False):
        """
        Clean the text, with the option to remove stop_words and to stem words
        """
        text = re.sub(r"ain't", " is not ", text.lower())
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"shan't", "shall not", text)
        text = re.sub(r"sha'n't", "shall not", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"let's", "let us", text)
        text = re.sub(r"how'd", "how did", text)
        text = re.sub(r"how'd'y", "how do you", text)
        text = re.sub(r"where'd", "where did", text)
        text = re.sub(r"'m", " am ", text)
        text = re.sub(r"'d", " would had ", text)
        text = re.sub(r"n\'t", " not ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"\'s", " is ", text)
        text = re.sub(r"\'cause", "because", text)
        text = re.sub(r"ma'am", "madam", text)
        text = re.sub(r"o'clock", "of the clock", text)
        text = re.sub(r"y'all", "you all", text)
        # remove hyper link
        text = re.sub(r"(\S*)https?://\S*", lambda m: m.group(1), text)

        text = re.sub(r"[^A-Za-z0-9]", " ", text)

        # number translate
        text = re.sub(r"1st", " first ", text)
        text = re.sub(r"2nd", " second ", text)
        text = re.sub(r"3rd", " third ", text)
        text = re.sub(r"4th", " fourth ", text)
        text = re.sub(r"5th", " fifth ", text)
        text = re.sub(r"6th", " sixth ", text)
        text = re.sub(r"7th", " seventh ", text)
        text = re.sub(r"8th", " eighth ", text)
        text = re.sub(r"9th", " ninth ", text)
        text = re.sub(r"10th", " tenth ", text)
        text = re.sub(r"0", " zero ", text)
        text = re.sub(r"1", " one ", text)
        text = re.sub(r"2", " two ", text)
        text = re.sub(r"3", " three ", text)
        text = re.sub(r"4", " four ", text)
        text = re.sub(r"5", " five ", text)
        text = re.sub(r"6", " six ", text)
        text = re.sub(r"7", " seven ", text)
        text = re.sub(r"8", " eight ", text)
        text = re.sub(r"9", " nine ", text)

        # symbol replacement
        text = re.sub(r"&", " and ", text)
        text = re.sub(r"\|", " or ", text)
        text = re.sub(r"=", " equal ", text)
        text = re.sub(r"\+", " plus ", text)
        text = re.sub(r"₹", " rs ", text)  # 测试！
        text = re.sub(r"\$", " dollar ", text)
        text = re.sub(r"&amp", " and ", text)
        text = re.sub(r"&quot", ' " ', text)
        text = re.sub(r"&lt", " less than ", text)
        text = re.sub(r"&gt", " greater than ", text)
        text = re.sub(r"&nbsp", " ", text)

        text = ''.join([c for c in text if c not in punctuation])
        text = re.sub(r"\s+", " ", text)

        # Optionally, remove stop words
        if remove_stop_words:
            text = text.split()
            text = [w for w in text if not w in stop_words]
            text = " ".join(text)

        # Optionally, shorten words to their stems
        if stem_words:
            text = text.split()
            stemmer = SnowballStemmer('english')
            stemmed_words = [stemmer.stem(word) for word in text]
            text = " ".join(stemmed_words)

        return str(text)

    @staticmethod
    def get_unigram_words(que):
        return [word for word in word_tokenize(que.lower()) if word not in stop_words]
