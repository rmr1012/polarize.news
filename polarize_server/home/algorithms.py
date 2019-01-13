#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import os
import numpy as np
import pandas as pd
import pprint
import nltk
from nltk.tag import pos_tag
nltk.download('averaged_perceptron_tagger')

from newsapi import NewsApiClient

import pickle
from sklearn.feature_extraction.text import CountVectorizer

# initialize connection to newsapi

newsapi = NewsApiClient(api_key='a3b76c5e036947daaa13d4aaf3acab5c')

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


def loadModel(path):
    data = pickle.load(open(path, 'rb'))
    return (data[0], data[1])


def get_binary_bias(inStr, model, vocab):
    """Predicts a binary bias value for each article."""

    cv1 = CountVectorizer(binary=True, vocabulary=vocab)

    cv1.fit([inStr])
    X1 = cv1.transform([inStr])
    return np.float(model.predict(X1) * 2 - 1)

realpath = os.path.dirname(__file__)
(model, vocab) = loadModel(os.path.join(realpath,'model.pk'))


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

    return fuzzy


def get_keywords(article, remove_duplicates=True, nouns_only=False):
    # text = article['title'] + ' ' + article['description']
    # content = article['content']

    # if content is not None:
    #     text += ' ' + content

    # p = re.compile(r"(\b[-']\b)|[\W_]")
    # text_to_analyze = p.sub(lambda m: (m.group(1) if m.group(1) else ' '
    #                         ), text)

    stacked = stack(article['title'], article['description'], article['content'])
    cleaned = clean(stacked)

    tagged = pos_tag(cleaned.split())

    if remove_duplicates:
        keywords = list(set([word.lower() for (word, pos) in tagged
                        if pos == 'NNP' or pos == 'VBG' or pos == 'VBD'
                        ]))
    elif remove_duplicates == False and nouns_only == False:
        keywords = list([word.lower() for (word, pos) in tagged if pos
                        == 'NNP' or pos == 'VBG' or pos == 'VBD'])
    elif remove_duplicates == False and nouns_only:
        noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']

        keywords = list([word.lower() for (word, pos) in tagged if pos in noun_tags])

    return keywords


def fast_sim(keywords, article):
    art_kw = get_keywords(article)

    score = 0
    for (i, kw) in enumerate(art_kw):
        if kw in keywords:
            score += 1

    return score


def get_similar_articles(article, search_articles, bias):
    kws = get_keywords(article)

    corr = fast_sim(kws, article)  # get autocorrelation to scale output.
    score = np.zeros_like(search_articles)

    for (i, sart) in enumerate(search_articles):
        if article == search_articles[i]:
            continue
        score[i] = fast_sim(kws, sart) / corr

    data = {'score': score, 'article': search_articles}
    df = pd.DataFrame.from_dict(data)

    df = df.sort_values('score', ascending=False).reset_index()

    i = 0
    while True:
        rawStr = stack(df['article'][i]['title'], df['article'
                       ][i]['description'], df['article'][i]['content'])
        cleanStr = clean(rawStr)
        pBias = get_binary_bias(cleanStr, model, vocab)[0]

        if bias == pBias:
            return df['article'][i]

        i += 1

        if i >= len(search_articles):
            print('Error: No similar articles with bias %f' % bias)
            return -1


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
    return REPLACE_NO_SPACE.sub(NO_SPACE, inStr.lower())


def get_most_common_keywords(articles, n):

    kw_lists = []

    if type(articles) != list:
        kw_lists.append(get_keywords(articles, remove_duplicates=False,
                        nouns_only=True))
    else:
        for article in articles:
            kw_lists.append(get_keywords(article,
                            remove_duplicates=False, nouns_only=True))

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


def get_headlines(threshold=0.01, page_size=10, sources=relevant_sources_str):
    """Called every thirty minutes."""

    newsapi = NewsApiClient(api_key='a3b76c5e036947daaa13d4aaf3acab5c')
    if sources == None:
        headlines = newsapi.get_top_headlines(language='en',
                country='us', page_size=10)
    else:
        headlines = newsapi.get_top_headlines(language='en',
                sources=sources, page_size=10)
    articles = headlines['articles']
    for (idx, article) in enumerate(articles):
        if article['title'] is None or article['description'] is None:
            del articles[idx]

  # get related articles

    first_article = articles[1]
    most_common_kws = ','.join(list(get_most_common_keywords([first_article],
            5).keys()))

    related_articles = newsapi.get_everything(language='en',
                                              q=most_common_kws[0],
                                              sort_by='relevancy',
                                              page_size=page_size)['articles']

    for (idx, rarticle) in enumerate(related_articles):
        if rarticle['title'] is None or rarticle['description'] is None \
            or rarticle['content'] is None:
            del related_articles[idx]

    kw0 = get_keywords(first_article)
    corr = fast_sim(kw0, first_article)

  # choose 3 dissimilar articles

    for (idx, rarticle) in enumerate(related_articles):
        related_articles[idx]['hash'] = str(hash(rarticle['title']))

        inStr = clean(stack(related_articles[idx]['title'],
                      related_articles[idx]['description'],
                      related_articles[idx]['content']))

        related_articles[idx]['bias'] = \
            get_fuzzy_bias(get_binary_bias(inStr, model, vocab),
                           related_articles[idx])

      # only for the first article

        related_articles[idx]['score'] = fast_sim(kw0, article) / corr

  # combine dictionary and create dataframe for easier processing

    rarticles_combined = {}
    for k in related_articles[0].keys():
        rarticles_combined[k] = tuple(a[k] for a in related_articles)

    df = pd.DataFrame.from_dict(rarticles_combined)
    df = df.sort_values('score', ascending=False).reset_index()

    left0 = []  # first content
    right0 = []  #
    inStr = clean(stack(first_article['title'],
                  first_article['description'], first_article['content'
                  ]))
    if get_fuzzy_bias(get_binary_bias(inStr, model, vocab),
                      first_article) < 0:
        left0.append(first_article)
    else:
        right0.append(first_article)

    hashes_used = []

    for i in range(0, len(df['score'])):
        if df['score'][i] > threshold and len(right0) <= 3 \
            and len(left0) <= 3:
            if df['bias'][i] < 0 and len(left0) < 3:
                left0.append(get_dict(df.iloc[i]))
                hashes_used.append(df.iloc[i]['hash'])
            elif df['bias'][i] > 0 and len(right0) < 3:
                right0.append(get_dict(df.iloc[i]))
                hashes_used.append(df.iloc[i]['hash'])
        else:
            break

    return [{"left":left0, "right":right0}]


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
    article = get_headlines(page_size=10, sources=relevant_sources_str)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(article)


if __name__ == '__main__':
    main()
