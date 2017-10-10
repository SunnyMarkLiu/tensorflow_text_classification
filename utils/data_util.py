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
from conf.configure import Configure

import re
from string import punctuation
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer

word_tokenize = WordPunctTokenizer().tokenize
stop_words = ['the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 'which', 'this', 'that', 'these',
              'those', 'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while',
              'during', 'to', 'What', 'Which', 'Is', 'If', 'While', 'This']


def load_text_datasets():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    :return: 
    """
    # Load data from files
    positive_examples = list(open(Configure.positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(Configure.negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [TextCleaner.clean_text(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def load_embedding_vectors_glove(vocabulary, glove_w2c_filename, vector_size):
    """
    load embedding_vectors from the pre-trained glove vectors
    :param vocabulary: 
    :param glove_w2c_filename: 
    :param vector_size: 
    :return: [vocabulary_size, embedding_size]
    """
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(glove_w2c_filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors


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
