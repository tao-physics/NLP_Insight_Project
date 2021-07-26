# import package

import transformers
from transformers import RobertaTokenizer

import tokenizers
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer

# use roberta
MAX_LEN = 150
ROBERTA_PATH = './input/roberta-base'
MODEL = 'roberta'

OUTPUT_SIZE_SENTIMENT = 3
OUTPUT_SIZE_LABEL = 14

OUTPUT_SIZE = OUTPUT_SIZE_SENTIMENT


# tokenizer needs a vocabulary
# TOKENIZER = BertWordPieceTokenizer(BERT_PATH + '/vocab.txt')

# max-len can be set in the parameters
# TOKENIZER = transformers.BertTokenizer(BERT_PATH + '/vocab.txt')
TOKENIZER = RobertaTokenizer.from_pretrained(ROBERTA_PATH)

# run the bert model
TRAIN_BATCH_SIZE = 24
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
EPOCHS = 5

# files
INPUT_PATH = './input'

TRAIN_FILE = INPUT_PATH + '/train.csv'
TEST_FILE = INPUT_PATH + '/test.csv'

TOTAL_FOLDS = 5
FOLD_COLS = 'Sentiment' # ['Label', 'Sentiment']
TRAIN_FOLDS_FILE = INPUT_PATH + '/train_folds.csv'

OUTPUT_PATH = './output'
SAVED_MODEL_PATH = './saved_model'