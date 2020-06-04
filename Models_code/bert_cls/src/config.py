# import package

import transformers
import tokenizers
from tokenizers import BertWordPieceTokenizer

# use bert
MAX_LEN = 150
BERT_PATH = './input/bert-base-uncased'

OUTPUT_SIZE_SENTIMENT = 3
OUTPUT_SIZE_LABEL = 15

# tokenizer needs a vocabulary
# TOKENIZER = BertWordPieceTokenizer(BERT_PATH + '/vocab.txt')

# max-len can be set in the parameters
TOKENIZER = transformers.BertTokenizer(BERT_PATH + '/vocab.txt')

# run the bert model
TRAIN_BATCH_SIZE = 24
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
EPOCHS = 3

# files
INPUT_PATH = './input'

TRAIN_FILE = INPUT_PATH + '/train.csv'
TEST_FILE = INPUT_PATH + '/test.csv'

TOTAL_FOLDS = 5
FOLD_COLS = 'Sentiment' # ['Label', 'Sentiment']
TRAIN_FOLDS_FILE = INPUT_PATH + '/train_folds.csv'

OUTPUT_PATH = './output'
SAVED_MODEL_PATH = './saved_model'