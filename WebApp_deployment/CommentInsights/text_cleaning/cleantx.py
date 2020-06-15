### for text cleaning ###

import nltk
nltk.download('punkt')

nltk.download('stopwords')
from nltk.corpus import stopwords

import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
HTML_TAG = re.compile('<.*?>')
STOPWORDS = set(stopwords.words('english'))

# Stemmer
snow = nltk.stem.SnowballStemmer('english')

lemmatizer = nltk.stem.WordNetLemmatizer() 


### text cleaning ###


def lower_text(text):
    text = text.lower()
    return text

# Whitespace include spaces, newlines \n and tabs \t , 
# and consecutive whitespace are processed together. 
def remove_space(text):
    text = text.strip()
    test = text.split()
    return " ".join(test)

def remove_punctuations(text):
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)
    return text

def remove_symbols(text):
    text = re.sub(BAD_SYMBOLS_RE, '', text)
    return text

def remove_htmltags(text):
    test = re.sub(HTML_TAG, ' ', text)
    return text

def remove_stopwords(text):
    text = ' '.join([w for w in text.split() if w not in STOPWORDS])
    return text

def text_stem(text):
    words = [snow.stem(word) for word in text.split()]
    text = ' '.join(words)
    return text

def text_lemmatization(text):
    words = [lemmatizer.lemmatize(word) for word in text.split()]
    text = ' '.join(words)
    return text

# text preprocessing
def text_prepare(text):
    text = lower_text(text)
    text = remove_punctuations(text)
    text = remove_symbols(text)
    text = remove_htmltags(text)
    text = remove_space(text)
    text = remove_stopwords(text)
    text = text_lemmatization(text)
    return text
