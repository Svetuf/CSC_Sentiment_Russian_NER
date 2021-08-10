import collections
import nltk
import numpy as np
import re
import spacy

from matplotlib import pyplot as plt
from nltk import ngrams
from nltk.probability import FreqDist
from pymystem3 import Mystem
from spacy.lang.ru import Russian


def read_file(file):
    f = open(file, 'r')
    lines = f.readlines()
    ans = []
    for line in lines:
        if '<column name=\"text\">' in line:
            ans.append(line.strip().replace('<column name=\"text\">', '').replace('</column>', ''))
    return ans


def regul(s, dat):
    ans = []
    for l in dat:
        z = re.findall(s, l)
        try:
            cur = l
            for i in z:
                cur = cur.replace(i, ' ')
            ans.append(cur)
        except:
            pass
    return ans


def create_hist(x, y, title, xlabel, ylabel):
    plt.hist(x, y)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()


def create_boxplot(s):
    plt.boxplot(s)

    plt.show()


def get_ngrams(n, data, best=True, worst=False, top=10):
    n_grams = []
    for i in data:
        n_grams += [a for a in ngrams(i.split(), n)]

    ngram_freq = collections.Counter(n_grams)
    if best == True:
        print(f'Лучшие {top} {n}-грамм')
        print(ngram_freq.most_common(10))
    if worst == True:
        print(f'Худшие {top} {n}-грамм')
        print(ngram_freq.most_common()[::-1][:10])


def get_frequent_words(data, top=10):
    split_it = data.split()
    counter = collections.Counter(split_it)
    return counter.most_common(top)


def get_lemmatize(text):
    m = Mystem()
    lemmas = m.lemmatize(text)
    return ''.join(lemmas)


def get_freq_pos(data, file):
    text = ' '.join(data)
    nlp = spacy.load(file)
    doc = nlp(text)

    s = []
    for token in doc:
        s.append((token.text, token.pos_))

    val_1 = collections.Counter(y for (_, y) in s)

    most_occur_tag = val_1.most_common()
    return most_occur_tag


def get_pos(data, file):
    ans = []
    nlp = spacy.load(file)

    for s in data:
        ans.append(' '.join(t.pos_ for t in nlp(s)))

    return ans
