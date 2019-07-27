#!/usr/bin/env python
# coding: utf-8

# In[37]:




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer


truth = pd.read_csv('sorting_hat_NLC_ground_truth.csv',header=None)

truth = truth.rename(columns={0: "quality", 1: "house"})

truth = truth.drop(2,axis=1)

train=truth

X = train.quality
y = train.house
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state = 42)



nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))


sentence = ["Structure Intelligent"]
df = pd.DataFrame(data=sentence)
df = df.rename(columns={0: "quality"})
inp =df.quality


nb.predict(inp)

train


