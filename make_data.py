import json
import requests
import re
from datetime import datetime
from datetime import timedelta
from newsapi import NewsApiClient
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from sklearn import linear_model
from sklearn import svm
import random
import nltk
from nltk.tag import pos_tag
nltk.download('averaged_perceptron_tagger')

newsapi = NewsApiClient(api_key='a3b76c5e036947daaa13d4aaf3acab5c')
classifiers = [
    #svm.SVR(),
    linear_model.SGDRegressor(),
    linear_model.BayesianRidge(),
    linear_model.LassoLars(),
    linear_model.ARDRegression(),
    linear_model.PassiveAggressiveRegressor(),
    linear_model.TheilSenRegressor(),
    linear_model.LinearRegression()]

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
    'the-hill':'The Hill',
    'the-huffington-post':'HuffPost',
    'the-new-york-times':'NY Times',
    'the-washington-post':'WashPost',
    'the-washington-times':'WashTimes',
    'time':'Time',
    'usa-today':'USA-Today',
    'vice-news':'Vice'
}

# removing un-credible news sources such as berbrit
bias={'abc-news':-.4, 'al-jazeera-english':-.25, 'associated-press':-.1, 'axios':-.25, 'cbs-news':-.35, 'cnn':-.65, 'fox-news':.65, 'google-news':-.5, 'msnbc':-.65, 'national-review':.7, 'nbc-news':-.45, 'newsweek':-.7, 'new-york-magazine':-.65, 'politico':-.1, 'reuters':-.1, 'the-american-conservative':.3, 'the-hill':-.2, 'the-huffington-post':-.6, 'the-new-york-times':-.35, 'the-washington-post':-.5, 'the-washington-times':.5, 'time':-.55, 'usa-today':-.3, 'vice-news':-.45}
def listSources():
    dat = json.load(open("sources.json","r"))
    bias_list=[]
    for source in dat["sources"]:
        if source["category"]=="general":
            bias_list.append(source["id"])
    return bias_list

def loadModel(path):
    data = pickle.load(open(path,"rb"))
    return data[0],data[1]
(model, vocab) = loadModel('model.pk')
def predictBias(inStr, model, vocab):
    cv1 = CountVectorizer(binary=True,vocabulary=vocab)
    cleanStr=REPLACE_NO_SPACE.sub(NO_SPACE, inStr.lower())
    cv1.fit([cleanStr])
    X1 = cv1.transform([cleanStr])
    return model.predict(X1)
def batchPredict(mat, model, vocab):
    cv1 = CountVectorizer(binary=True,vocabulary=vocab)
    cv1.fit(mat)
    X1 = cv1.transform(mat)
    return model.predict(X1)
def getRandFromSource(source):
    fromDate=datetime.now()-timedelta(days=random.randint(1,30))
    toDate=fromDate-timedelta(days=1)
    articles=[]
    print("Getting 1 News from "+source+" on "+(fromDate).strftime("%Y-%m-%d"))
    all_articles = newsapi.get_everything(#q='bitcoin',
                                  sources=source,
                                  from_param=fromDate.strftime("%Y-%m-%d"),
                                  to=toDate.strftime("%Y-%m-%d"),
                                  language='en',
                                  page_size=100
                                  )
    article= all_articles["articles"][random.randint(0,len(all_articles["articles"]))]
    rawStr=stack(article["title"] , article["description"] , article["content"])
    return clean(rawStr)

def stack(title,desc,content):
    return content+desc*2+title*3
def clean(inStr):
    REPLACE_NO_SPACE = \
        re.compile('(\xe2\x80\xa6)|(\\[+.* chars\\])|(\r)|(\n)|(\\.)|(\\;)|(\\:)|(\\!)|(\')|(\\?)|(\\,)|(")|(\\()|(\\))|(\\[)|(\\])|(\\d+)')
    NO_SPACE = ''
    SPACE = ' '
    txt=REPLACE_NO_SPACE.sub(NO_SPACE, inStr.lower())
    s=set(stopwords.words('english'))
    return ' '.join(list(filter(lambda w: not w in s,txt.split())))
relevant_sources = [
    #'abc-news',
    #'al-jazeera-english',
    #'associated-press',
    #'axios',
    'cbs-news',
    'cnn',
    'fox-news',
    #'google-news',
    'msnbc',
    'national-review',
    'nbc-news',
    'newsweek',
    'new-york-magazine',
    'politico',
    'reuters',
    'the-american-conservative',
    'the-hill',
    #'the-huffington-post',
    'the-new-york-times',
    'the-washington-post',
    'the-washington-times',
    'time',
    'usa-today',
    #'vice-news',
    ]

relevant_sources_str = ','.join(relevant_sources)


def get_headlines(threshold=0.12, page_size=10, sources=relevant_sources_str,topic='white house'):
    """Called every thirty minutes."""
    newsapi = NewsApiClient(api_key='a3b76c5e036947daaa13d4aaf3acab5c')
    if sources == None:
        headlines = newsapi.get_everything(language='en', q=topic,
        country='us', page_size=page_size)
    else:
        headlines = newsapi.get_everything(language='en', q=topic,
        sources=sources, page_size=page_size)
    articles = headlines['articles']
    completeArticles=[]
    for (idx, article) in enumerate(articles):
        if article['title'] is not None and article['description'] is not None and article['content'] is not None:
            print("adding"+str(idx))
            completeArticles.append(article)
  # get related articles
    #print(articles)
    cleanArticles=[]
    s=set(stopwords.words('english'))
    for index,article in enumerate(completeArticles):
        print("clenaing"+str(index))
        cleanArticles.append(clean(stack(article['title'],article['description'],article['content'])))

    #print(cleanArticles)
    cleanArticleBias=batchPredict(cleanArticles,model,vocab)

    print(cleanArticleBias)


    vect = TfidfVectorizer(min_df=1)
    print("tic")
    tfidf = vect.fit_transform(cleanArticles)
    print("toc")
    corr=(tfidf * tfidf.T).A
    print(corr)
    avalStatus=[1]*len(cleanArticles)
    clusters=[]
    for count,articleCorr in enumerate(corr):
        if(avalStatus[count]):
            avalStatus[count]=0
            idxs=sorted(range(len(articleCorr)), key=lambda i: articleCorr[i])[::-1][1:]
            idxAboveThresh=[count]
            for idx in idxs:
                if(articleCorr[idx]>=threshold):
                    if avalStatus[idx]:
                        idxAboveThresh.append(idx)
                        avalStatus[idx]=0
            print(idxAboveThresh)
            if(len(idxAboveThresh)>2): ## more than just itself
                clusters.append(idxAboveThresh)
            elif(len(idxAboveThresh)==2 and cleanArticleBias[count] != cleanArticleBias[idxAboveThresh[0]]):
                clusters.append(idxAboveThresh)

    print(avalStatus)
    print("^ shoulld be all 0")
    print(clusters)

    cardRack=[]

    for cluster in clusters:
        leftArray=[]
        rightArray=[]
        for idx in cluster:
            article=completeArticles[idx]
            article["source"]=TransName[article["source"]["id"]]
            if cleanArticleBias[idx]: # if conservative
                rightArray.append(article)
            else:
                leftArray.append(article)
        cardRack.append({"left":leftArray,"right":rightArray})

    return cardRack

        #rank corrs

def getAllFromSource(source,iterMax=1,dateGap=15):
    # url="https://newsapi.org/v2/everything?sources="+source+"&apiKey=9626af5c1fac45dbb0363c8aac416905"
    # response = requests.get(url)

    fromDate=datetime.now()
    toDate=fromDate-timedelta(days=dateGap)
    articles=[]
    print("Getting "+str(iterMax*100)+" News from "+source+" from now to "+(fromDate-timedelta(days=dateGap*iterMax)).strftime("%Y-%m-%d"))
    for i in tqdm(range(iterMax)):
        fromDate=fromDate-timedelta(days=dateGap)
        toDate=fromDate-timedelta(days=dateGap)
        all_articles = newsapi.get_everything(#q='bitcoin',
                                      sources=source,
                                      from_param=fromDate.strftime("%Y-%m-%d"),
                                      to=toDate.strftime("%Y-%m-%d"),
                                      language='en',
                                      page_size=100
                                      )
        for article in all_articles["articles"]:
            try:
                rawStr=stack(article["title"] , article["description"] , article["content"])
                articles.append(clean(rawStr))
            except:
                pass
                                      # sort_by='relevancy'                                  #page=100)
    return articles

def pullData():
    truth=[]
    articles=[]
    for source in list(bias.keys()):
        freshArticles=getAllFromSource(source,iterMax=10,dateGap=3)
        articles += freshArticles
        truth+= [bias[source]]*len(freshArticles)
    json.dump(truth,open("truth.json","w"))
    json.dump(articles,open("articles.json","w"))

def trainModel():
    truth=json.load(open("truth.json","r"))
    evenout=lambda val: 1 if val>0 else 0
    intTruth=[evenout(val) for val in truth]
    articles=json.load(open("articles.json","r"))

    cv = CountVectorizer(binary=True)
    cv.fit(articles)
    X = cv.transform(articles)
    vocab=cv.get_feature_names()

    X_train, X_val, y_train, y_val = train_test_split(
        X, intTruth, train_size = 0.75)

    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        lr = LogisticRegression(C=c)
        lr.fit(X_train, y_train)
        print ("Accuracy is %s" % (accuracy_score(y_val, lr.predict(X_val))))

    final_model = LogisticRegression(C=0.5)
    final_model.fit(X, intTruth)
    return final_model ,cv, vocab
def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.

    get_top_n_words(["I love Python", "Python is a language programming", "Hello world", "I love the world"]) ->
    [('python', 2),
     ('world', 2),
     ('love', 2),
     ('hello', 1),
     ('is', 1),
     ('programming', 1),
     ('the', 1),
     ('language', 1)]
    """
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# articles=json.load(open("articles.json","r"))
# cars_for_sell = articles
# common_words = get_top_n_words(cars_for_sell, 20)
# for word, freq in common_words:
#     print(word, freq)


def get_fuzzy_bias(bias, article):
  text = article['title'] + ' ' + article['description']
  content = article['content']

  if content is not None:
      text += ' ' + content

  p = re.compile(r"(\b[-']\b)|[\W_]")
  text_to_analyze = p.sub(lambda m: (m.group(1) if m.group(1) else " "), text)

  tagged = pos_tag(text_to_analyze.split())

  noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
  adj_tags = ['JJ', 'JJR', 'JJS']

  nouns = list(set([word.lower() for word,pos in tagged if pos in noun_tags]))
  adj = list(set([word.lower() for word,pos in tagged if pos in adj_tags]))

  fuzzy = len(adj)/len(nouns)

  return bias*fuzzy

def extreames():
    feature_to_coef = {word: coef for word, coef in zip(vocab, model.coef_[0])}
    for best_positive in sorted(feature_to_coef.items(),key=lambda x: x[1], reverse=True)[:5]:
        print (best_positive)
    for best_negative in sorted(feature_to_coef.items(),key=lambda x: x[1])[:5]:
        print (best_negative)

# articles=json.load(open("articles.json","r"))
# longstr=' '.join(articles)
# text_to_analyze = longstr
# tagged = pos_tag(text_to_analyze.split())
#
# noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
# adj_tags = ['JJ', 'JJR', 'JJS']
#
# nouns = list(set([word.lower() for (word, pos) in tagged if pos
#              in noun_tags]))
# adj = list(set([word.lower() for (word, pos) in tagged if pos
#            in adj_tags]))
# cv=CountVectorizer().fit(nouns)
# X = cv.transform(nouns)
# words_freq = [(word, sum_words[0, idx]) for word, idx in     cv.vocabulary_.items()]
# words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
