# text classification in tensorflow
Implementing multi models for Text Classification in TensorFlow.

## Contents
### Data and Preprocess
#### Data
Models are used to perform sentiment analysis on movie reviews from the [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/), which contains 25,000 highly polar movie reviews for training, and 25,000 for testing.<br/>
In this task, given a movie review, the model attempts to predict whether it is positive or negative. This is a binary classification task.

#### Preprocess
1. Load positive and negative sentences from the raw data files.
2. Clean the text data.
3. Pad each sentence to the maximum sentence length.
4. Word vector mapping, Each sentence becomes a bag of word vectors.

### Models
#### 1. FastText
![](./imgs/fast_text_model.png)
- word representations: [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)
- text classification: [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)

#### 2. TextCNN
![](./imgs/text_cnn_model.png)
![](./imgs/text_cnn_model_explain.png)
- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1510.03820)

### Results
- train/valid/test size = 30908/3434/8585
- glove word vector embedding dims = 300
- max_document_length = 200
- max_learning_rate = 0.01, decay_rate = 0.8, decay_steps = 2000
- epochs = 10
- embedding_trainable = False

| Models     | FastText | TextCNN |
| :----:     | :----:   | :----: |
| Accuracy   | 0.832848 |  |
| Training   |  48.94s  |  |

- embedding_trainable = True

| Models     | FastText | TextCNN |
| :----:     | :----:   | :----: |
| Accuracy   | 0.854397 |  |
| Training   | 2352.95s |  |

## References
- [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)
- [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1510.03820)
- [LSTM Networks for Sentiment Analysis](http://deeplearning.net/tutorial/lstm.html)
- [Github: cnn-text-classification-tf](https://github.com/cahya-wirawan/cnn-text-classification-tf)
- [基于 word2vec 和 CNN 的文本分类 ：综述 & 实践](https://zhuanlan.zhihu.com/p/29076736)

## License
This project is licensed under the terms of the MIT license.
