#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import os
import numpy as np
import pandas as pd
import pprint
import nltk
from nltk.tag import pos_tag
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from sklearn.feature_extraction.text import TfidfVectorizer

from newsapi import NewsApiClient

import pickle
from sklearn.feature_extraction.text import CountVectorizer

# initialize connection to newsapi
newsapi = NewsApiClient(api_key='a3b76c5e036947daaa13d4aaf3acab5c')

hashes_used = []

relevant_sources = [
    'abc-news',
    'al-jazeera-english',
    'associated-press',
    'axios',
    'cbs-news',
    'cnn',
    'fox-news',
    'google-news',
    'msnbc',
    'national-review',
    'nbc-news',
    'newsweek',
    'new-york-magazine',
    'politico',
    'reuters',
    'the-american-conservative',
    'the-hill',
    'the-huffington-post',
    'the-new-york-times',
    'the-washington-post',
    'the-washington-times',
    'time',
    'usa-today',
    'vice-news',
    ]

relevant_sources_str = ','.join(relevant_sources)

TransName = {
    'abc-news':'ABC',
    'al-jazeera-english':'ABC',
    'associated-press':'Associated',
    'axios':'Axios',
    'cbs-news':'CBS',
    'cnn':'CNN',
    'fox-news':'FOX',
    'google-news':'Google',
    'msnbc':'MSNBC',
    'national-review':'National',
    'nbc-news':'NBC',
    'newsweek':'NewsWeek',
    'new-york-magazine':'NY-Mag',
    'politico':'Politico',
    'reuters':'Reuters',
    'the-american-conservative':'American-Cons',
    'the-hill':'The-Hill',
    'the-huffington-post':'HuffPost',
    'the-new-york-times':'NY-Times',
    'the-washington-post':'WashPost',
    'the-washington-times':'WashTimes',
    'time':'Time',
    'usa-today':'USA-Today',
    'vice-news':'Vice',
    'the-economist': 'Economist',
    'mashable': 'Mashable'
}


def loadModel(path):
    data = pickle.load(open(path, 'rb'))
    return (data[0], data[1])


def get_binary_bias(inStr, model, vocab):
    """Predicts a binary bias value for each article."""

    cv1 = CountVectorizer(binary=True, vocabulary=vocab)

    cv1.fit([inStr])
    X1 = cv1.transform([inStr])
    return np.float(model.predict(X1)*2-1)

realpath = os.path.dirname(__file__)
(model, vocab) = loadModel(os.path.join(realpath,'model.pk'))

def get_abs_fuzzy_bias(article):
    """Calculates a fuzzy bias value from a binary bias value and the article
    keywords.

    Parameters
    ----------
    bias : int (-1,1)
    article : dict

    Returns
    -------
    fuzzy_bias : float, typically between -1 and 1
    note: can exceed mag(1) if instances of adj > nouns
    """

    text = article['title'] + ' ' + article['description']
    content = article['content']

    if content is not None:
        text += ' ' + content

    p = re.compile(r"(\b[-']\b)|[\W_]")
    text_to_analyze = p.sub(lambda m: (m.group(1) if m.group(1) else ' '
                            ), text)

    tagged = pos_tag(text_to_analyze.split())

    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
    adj_tags = ['JJ', 'JJR', 'JJS']

    nouns = list(set([word.lower() for (word, pos) in tagged if pos
                 in noun_tags]))
    adj = list(set([word.lower() for (word, pos) in tagged if pos
               in adj_tags]))

    fuzzy = len(adj) / len(nouns)

    return fuzzy

def get_fuzzy_bias(bias, article):
    """Calculates a fuzzy bias value from a binary bias value and the article
    keywords.

    Parameters
    ----------
    bias : int (-1,1)
    article : dict

    Returns
    -------
    fuzzy_bias : float, typically between -1 and 1
    note: can exceed mag(1) if instances of adj > nouns
    """

    text = article['title'] + ' ' + article['description']
    content = article['content']

    if content is not None:
        text += ' ' + content

    p = re.compile(r"(\b[-']\b)|[\W_]")
    text_to_analyze = p.sub(lambda m: (m.group(1) if m.group(1) else ' '
                            ), text)

    tagged = pos_tag(text_to_analyze.split())

    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
    adj_tags = ['JJ', 'JJR', 'JJS']

    nouns = list(set([word.lower() for (word, pos) in tagged if pos
                 in noun_tags]))
    adj = list(set([word.lower() for (word, pos) in tagged if pos
               in adj_tags]))

    fuzzy = len(adj) / len(nouns)

    return fuzzy*bias


def get_keywords(article, remove_duplicates=True, nouns_only=False):
    stacked = stack(article['title'], article['description'], article['content'])
    cleaned = clean(stacked)

    tagged = pos_tag(cleaned.split())
    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']

    if remove_duplicates and nouns_only:
        keywords = list(set([word.lower() for (word, pos) in tagged
                        if pos in noun_tags]))
    elif remove_duplicates and not(nouns_only):
        keywords = list(set([word.lower() for (word, pos) in tagged
                        if pos in noun_tags or pos == 'VBG' or pos == 'VBD']))
    elif not(remove_duplicates) and not(nouns_only):
        keywords = list([word.lower() for (word, pos) in tagged if pos in noun_tags \
                        or pos == 'VBG' or pos == 'VBD'])
    elif not(remove_duplicates) and nouns_only:
        keywords = list([word.lower() for (word, pos) in tagged if pos in noun_tags])

    return keywords


def fast_sim(keywords, article):
    art_kw = get_keywords(article, remove_duplicates=True)

    score = 0
    for kw in art_kw:
        if kw in keywords:
            score += 1

    return score


def stack(title, desc, content):
    text = 3 * title + ' ' + 2 * desc
    if content is not None:
        text += ' ' + content

    return text


def clean(inStr):
    REPLACE_NO_SPACE = \
        re.compile('(\xe2\x80\xa6)|(\\[+.* chars\\])|(\r)|(\n)|(\\.)|(\\;)|(\\:)|(\\!)|(\')|(\\?)|(\\,)|(")|(\\()|(\\))|(\\[)|(\\])|(\\d+)'
                   )
    NO_SPACE = ''
    SPACE = ' '
    txt=REPLACE_NO_SPACE.sub(NO_SPACE, inStr.lower())
    s=set(stopwords.words('english'))
    return ' '.join(list(filter(lambda w: not w in s,txt.split())))




def get_most_common_keywords(articles, n):
    kw_lists = []

    if type(articles) != list:
        kw_lists.append(get_keywords(articles, remove_duplicates=False, nouns_only=True))
    else:
        for article in articles:
            kw_lists.append(get_keywords(article, remove_duplicates=False, nouns_only=True))

  # count frequency

    freq = {}
    for kws in kw_lists:
        for kw in kws:
            try:
                freq[kw] += 1
            except KeyError:
                freq[kw] = 1

    banned_kws = ['news']
    for bkw in banned_kws:
        try:
            del freq[bkw]
        except KeyError:
            continue

    df = pd.Series(freq, index=freq.keys())
    df = df.sort_values(ascending=False)

    return df[0:n]


def get_headlines(topic, threshold=0.02, page_size=10, sources=relevant_sources_str):
    """Called every thirty minutes."""
    newsapi = NewsApiClient(api_key='a3b76c5e036947daaa13d4aaf3acab5c')
    if sources == None:
        headlines = newsapi.get_top_headlines(language='en', country='us', page_size=10)
    else:
        headlines = newsapi.get_top_headlines(language='en', sources=sources, page_size=10)

    headlines = newsapi.get_everything(language='en', sort_by='relevancy', q=topic,
                                       sources=sources, page_size=page_size)

    articles = headlines['articles']

    for idx, article in enumerate(articles):
        if article['title'] is None or article['description'] is None or article['content'] is None:
            del article
            continue

    left = []  # first content
    right = []  #

    for idx, article in enumerate(articles):
        inStr = clean(stack(article['title'],
                            article['description'],
                            article['content']))

        bias = get_fuzzy_bias(get_binary_bias(inStr, model, vocab), article)
        hash_ = hash(article['title'])
        articles[idx]['bias'] = bias
        articles[idx]['hash'] = hash_

        if bias < 0 and hash_ not in hashes_used:
            left.append(article)
        elif bias > 0 and hash_ not in hashes_used:
            right.append(article)
    
        hashes_used =.append(hash_)


    # clean up some of the data
    for i in range(0,len(left)):
        left[i]['bias'] = np.abs(left[i]['bias'])
        left[i]['source'] = TransName[left[i]['source']['id']]
        left[i]['image'] = left[i]['urlToImage']
        left[i]['hash'] = str(hash(left[i]["title"]))[:-6]

    for i in range(0,len(right)):
        right[i]['bias'] = np.abs(right[i]['bias'])
        right[i]['source'] = TransName[right[i]['source']['id']]
        right[i]['image'] = right[i]['urlToImage']
        right[i]['hash'] = str(hash(right[i]["title"]))[:-6]

    return {"left":left[0:3], "right":right[0:3]}


def get_dict(series):
    d = {
        'source': series['source']['name'],
        'title': series['title'],
        'image': series['urlToImage'],
        'description': series['description'],
        'url': series['url'],
        'hash': series['hash'],
        'bias': series['bias'],
        }

    return d


relevant_sources = [
    'abc-news',
    'al-jazeera-english',
    'associated-press',
    'axios',
    'cbs-news',
    'cnn',
    'fox-news',
    'google-news',
    'msnbc',
    'national-review',
    'nbc-news',
    'newsweek',
    'new-york-magazine',
    'politico',
    'reuters',
    'the-american-conservative',
    'the-hill',
    'the-huffington-post',
    'the-new-york-times',
    'the-washington-post',
    'the-washington-times',
    'time',
    'usa-today',
    'vice-news',
    ]

relevant_sources_str = ','.join(relevant_sources)

def main():
    # article = get_headlines(page_size=10, sources=relevant_sources_str)
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(article)
    pass


if __name__ == '__main__':
    main()
