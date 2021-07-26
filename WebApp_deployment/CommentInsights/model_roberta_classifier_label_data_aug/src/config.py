# import package

import transformers
from transformers import RobertaTokenizer

import tokenizers
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer


ROOT_PATH = './model_roberta_classifier_label_data_aug'
INPUT_PATH = ROOT_PATH + '/input'

# use roberta
MAX_LEN = 150
ROBERTA_PATH = INPUT_PATH + '/roberta-base'
MODEL = 'roberta'

OUTPUT_SIZE_SENTIMENT = 3
OUTPUT_SIZE_LABEL = 14

OUTPUT_SIZE = OUTPUT_SIZE_LABEL


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


TRAIN_FILE = INPUT_PATH + '/train_2aug.csv'
TEST_FILE = INPUT_PATH + '/test.csv'

TOTAL_FOLDS = 5
FOLD_COLS = 'Label' # ['Label', 'Sentiment']
TRAIN_FOLDS_FILE = INPUT_PATH + '/train_folds_2aug.csv'

OUTPUT_PATH = ROOT_PATH + '/output'
SAVED_MODEL_PATH = ROOT_PATH + '/saved_model/model_2aug'



