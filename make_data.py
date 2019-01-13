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
import numpy as np
from sklearn import linear_model
from sklearn import svm
import random
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



# removing un-credible news sources such as berbrit
bias={'abc-news':-.4, 'al-jazeera-english':-.25, 'associated-press':-.1, 'axios':-.25, 'cbs-news':-.35, 'cnn':-.65, 'fox-news':.65, 'google-news':-.5, 'msnbc':-.65, 'national-review':.7, 'nbc-news':-.45, 'newsweek':-.7, 'new-york-magazine':-.65, 'politico':-.1, 'reuters':-.1, 'the-american-conservative':.3, 'the-hill':-.2, 'the-huffington-post':-.6, 'the-new-york-times':-.35, 'the-washington-post':-.5, 'the-washington-times':.5, 'time':-.55, 'usa-today':-.3, 'vice-news':-.45}
def listSources():
    dat = json.load(open("sources.json","r"))
    bias_list=[]
    for source in dat["sources"]:
        if source["category"]=="general":
            bias_list.append(source["id"])
    return bias_list
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
    REPLACE_NO_SPACE = re.compile("(…)|(\[+.* chars\])|(\r)|(\n)|(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
    NO_SPACE = ""
    SPACE = " "
    return REPLACE_NO_SPACE.sub(NO_SPACE, inStr.lower())

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

articles=json.load(open("articles.json","r"))
cars_for_sell = articles
common_words = get_top_n_words(cars_for_sell, 20)
for word, freq in common_words:
    print(word, freq)

def loadModel(path):
    data = pickle.load(open(path,"rb"))
    return data[0],data[1]
def predictBias(inStr, model, vocab):
    cv1 = CountVectorizer(binary=True,vocabulary=vocab)
    cleanStr=REPLACE_NO_SPACE.sub(NO_SPACE, inStr.lower())
    cv1.fit([cleanStr])
    X1 = cv1.transform([cleanStr])
    return model.predict(X1)
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
