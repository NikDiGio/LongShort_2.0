## Pre-processing steps:
# - Tokenization
# - Stop-words removal
# - Word Lemmatization
# - Left padding

import pandas as pd
import json
import spacy
import nltk
nltk.download("stopwords")
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.corpus import stopwords

punctuations="?:!.,;"

wordnet_lemmatizer = WordNetLemmatizer()


def Stem_Stopwords(sentence):
    cachedStopWords = stopwords.words("english")  # caching the stopwords instance gives a ~70x speedup
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
      if word in punctuations or word in cachedStopWords:
          token_words.remove(word)
    for word in token_words:
        stem_sentence.append(wordnet_lemmatizer.lemmatize(word, 'v')) # POS -> Part Of Speech "v" for verbs
        stem_sentence.append(" ")
    return stem_sentence




def Apply_Stem_Stopwords(df, col_name):
    new_sent = []
    for i in range(0, df.shape[0]):
        new_sent.append(Stem_Stopwords(df[col_name].iloc[i]))
    df[col_name] = new_sent
    return df



def LeftPadding(df, col_name):
    max_len = max(df[col_name].apply(len))
    new_data = [['</s>'] * (max_len - len(l)) + l for l in df[col_name]]  # '</s>' suggested by nltk.pad_sequence()
    return new_data



def WrapFunction(df, col_name):
    df = Apply_Stem_Stopwords(df, col_name)
    df[col_name] = LeftPadding(df, col_name)
    df[col_name] = df[col_name].apply("".join)
    return df

def json_to_pandas_converted(path):
    f = open(path)
    data = json.load(f)
    df = pd.DataFrame(data)
    col_names = pd.DataFrame(data).columns.values
    sent_value = []
    if 'sentiment' in col_names:
        df['sentiment'] = df['sentiment'].astype(float)
        for score in df['sentiment']:
            if score > 0.:
                sent_value.append(2)
            elif score == 0.:
                sent_value.append(1)
            else:
                sent_value.append(0)
    elif 'sentiment score' in col_names:
        df['sentiment score'] = df['sentiment score'].astype(float)
        df['spans'] = df['spans'].apply(" ".join)
        for score in df['sentiment score']:
            if score > 0.:
                sent_value.append(2)
            elif score == 0.:
                sent_value.append(1)
            else:
                sent_value.append(0)

    df['label'] = sent_value
    return df




