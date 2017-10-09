# tensorFlow_text_classification
Implementing multi models for Text Classification in TensorFlow.

## Contents
### Data and Preprocess
#### Data
Models are used to perform sentiment analysis on movie reviews from the [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/), which has a set of 25,000 highly polar movie reviews for training, and 25,000 for testing.<br/>
In this task, given a movie review, the model attempts to predict whether it is positive or negative. This is a binary classification task.

#### Preprocess
1. Load positive and negative sentences from the raw data files.
2. Clean the text data.
3. Pad each sentence to the maximum sentence length. We append special <PAD> tokens to all other sentences to make them 59 words. Padding sentences to the same length is useful because it allows us to efficiently batch our data since each example in a batch must be of the same length.
4. Wordvector mapping, Each sentence becomes a bag of word vectors.

### Models
#### FastText
1. word representations: Enriching Word Vectors with Subword Information
2. text classification: Bag of Tricks for Efficient Text Classification

### Results
| Models     | FastText | right |
| :----:     | :----:   | :----: |
| Train Time |  |  |
| Accuracy   |  |  |

## References
- [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)
- [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1510.03820)
- [LSTM Networks for Sentiment Analysis](http://deeplearning.net/tutorial/lstm.html)

## License
This project is licensed under the terms of the MIT license.
