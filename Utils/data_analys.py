import csv
import pandas as pd
import spacy
import re
from collections import Counter
from nltk import ngrams
from nltk.tokenize import TweetTokenizer


DATA_DIR = '../../Data/Russian Twitter Corpus/'


def read_csv_clear(name):
    '''
    Читает данные из файла svc, и преобразует их для дальнейшей обработки
    :param name: имя файла
    :return: dataframe с подготовленными данными
    '''
    x = pd.read_csv(f"{DATA_DIR}" + name, sep=';', header=None, engine='c')
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    for i in range(12):
        if i != 3:
            del x[i]
    for i,_ in enumerate(x[3]):
        x.at[i, 3] = ' '.join(tknzr.tokenize(x.at[i, 3]))

    x['size'] = pd.Series(len(x[3]), index=x.index)
    for i, _ in enumerate(x['size']):
        x.at[i, 'size'] = len(x.at[i, 3])

    reg = re.compile('[^a-zA-Z ][^а-яА-Я ]]')
    for i,_ in enumerate(x[3]):
        x.at[i, 3] = reg.sub('', x.at[i, 3])

    return x


def create_ngrams_dictionary(bigrams_list):
    '''
    Вычисляет словарь из списка n-грамм
    :param bigrams_list: лист n-грамм
    :return: Counter n-грамм
    '''
    return Counter([elem for lst in bigrams_list for elem in lst])


def calculate_ngrams(x, n=2):
    '''
    Вычилсяет n-граммы
    :param x: dataframe с данными
    :param n: параметр
    :return: словарь n-грамм
    '''
    bigrams_list = []
    for i, _ in enumerate(x[3]):
        bigrams_list.append(list(ngrams((y for y in f'^ {x.at[i, 3]} $'.split(' ') if y != ''), n)))

    return create_ngrams_dictionary(bigrams_list)


def POS_bigrams(x, n=2):
    '''
    Вычисляет n-граммы частей речи
    :param x: dataframe с данными
    :param n: параметр
    :return: словарь n-грамм
    '''
    nlp = spacy.load("ru_core_news_lg")
    bigrams_list = []
    for i, _ in enumerate(x[3]):
        words = [token.pos_ for token in nlp(x.at[i, 3])]
        bigrams_list.append(list(ngrams(words, n)))

    return create_ngrams_dictionary(bigrams_list)
