import csv
import pandas as pd
from collections import Counter
from nltk import ngrams
from nltk.tokenize import TweetTokenizer
import spacy


def read_csv_clear(name):
    x = pd.read_csv("../../Data/Russian Twitter Corpus/" + name, sep=';', header=None, engine='c')
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    for i in range(12):
        if i != 3:
            del x[i]
    for i in range(len(x[3])):
        x.at[i, 3] = ' '.join(tknzr.tokenize(x.at[i, 3]))

    x['size'] = pd.Series(len(x[3]), index=x.index)
    for i in range(len(x['size'])):
        x.at[i, 'size'] = len(x.at[i, 3])

    return x


def calculate_ngrams(x, n=2):
    bigramsList = []
    for i in range(len(x['size'])):
        bigramsList.append(list(ngrams(('^' + x.at[i, 3] + '$').split(' '), n)))

    bigramsDist = Counter()
    for lst in bigramsList:
        for elem in lst:
            bigramsDist.update({str(elem): 1})
    return bigramsDist


def POS_bigrams(x, n=2):
    nlp = spacy.load("ru_core_news_lg")
    bigramsList = []
    for i in range(len(x[3])):
        doc = nlp(x.at[i, 3])
        words = []
        for token in doc:
            words.append(token.pos_)
        bigramsList.append(list(ngrams(words, n)))

    bigramsDist = Counter()
    for lst in bigramsList:
        for elem in lst:
            bigramsDist.update({str(elem): 1})
    return bigramsDist
