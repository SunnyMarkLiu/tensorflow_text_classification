# BiLSTM
## Parameters
```
usage: bi_lstm_train.py --help

optional arguments:
  -h, --help            show this help message and exit
  --test_split_percentage TEST_SPLIT_PERCENTAGE
                        Percentage of the training data to use for validation
  --validate_split_percentage VALIDATE_SPLIT_PERCENTAGE
                        Percentage of the training data to use for validation
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of word embedding (default: 300)
  --max_document_length MAX_DOCUMENT_LENGTH
                        Max document length (default: 200)
  --lstm_drop_out [LSTM_DROP_OUT]
                        Use dropout with LSTM (default: False)
  --nolstm_drop_out
  --lstm_input_keep_prob LSTM_INPUT_KEEP_PROB
                        Dropout keep probability about LSTM input(default:
                        1.0)
  --lstm_output_keep_prob LSTM_OUTPUT_KEEP_PROB
                        Dropout keep probability about LSTM output(default:
                        1.0)
  --lstm_state_keep_prob LSTM_STATE_KEEP_PROB
                        Dropout keep probability about LSTM state(default:
                        1.0)
  --embedding_trainable [EMBEDDING_TRAINABLE]
                        Word embedding trainable (default: False)
  --noembedding_trainable
  --max_learning_rate MAX_LEARNING_RATE
                        Max learning_rate when start training (default: 0.01)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --epochs EPOCHS       Number of training epochs (default: 200)
  --train_verbose_every_steps TRAIN_VERBOSE_EVERY_STEPS
                        Show the training info every steps (default: 100)
  --evaluate_every_steps EVALUATE_EVERY_STEPS
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every_steps CHECKPOINT_EVERY_STEPS
                        Save model after this many steps (default: 100)
  --max_num_checkpoints_to_keep MAX_NUM_CHECKPOINTS_TO_KEEP
                        Number of checkpoints to store (default: 5)
  --decay_rate DECAY_RATE
                        Learning rate decay rate (default: 0.9)
  --decay_steps DECAY_STEPS
                        Perform learning rate decay step (default: 10000)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regulaization rate (default: 10000)
  --log_message LOG_MESSAGE
                        log dir message (default: timestamp)
```
## Train
```
python bi_lstm_train.py
```

## Predict
```
python bi_lstm_predict.py --checkpoint_dir run/2017_12_08_13_30_39/checkpoints/
```
