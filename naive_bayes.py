from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import pandas as pd
import pickle

print("Reading files")
train = pd.read_csv('../data/train_fasttext_reliable_norm.csv',header=None,names=['title'])
val = pd.read_csv('../data/val_fasttext_reliable_norm.csv',header=None,names=['title'])
test = pd.read_csv('../data/test_fasttext.csv',header=None,names=['title'])

print("Formatting dataframes")
train['category'] = train['title'].apply(lambda x: ''.join(x.split()[0][9:]))
train['title'] = train['title'].apply(lambda x: ' '.join(x.split()[1:]))
val['category'] = val['title'].apply(lambda x: ''.join(x.split()[0][9:]))
val['title'] = val['title'].apply(lambda x: ' '.join(x.split()[1:]))

y_train = train['category']
y_val = val['category']

print("Vectorizing features")
count_vect = CountVectorizer()
X_train_vect = count_vect.fit_transform(train['title'] )
X_test_vect = count_vect.transform(val['title'])

print("Training Multinomial Naive Bayes")
clf = MultinomialNB()
clf.fit(X_train_vect, y_train)

print("Predicting Train")
yTrainPredict = clf.predict(X_train_tfidf)
print("Predicting Validation")
yValPrediction = clf.predict(X_val_tfidf)
print("Balanced Accuracy Score: %.2f" % balanced_accuracy_score(y_train, yTrainPredict))
print("Balanced Accuracy Score: %.2f" % balanced_accuracy_score(y_val, yValPrediction))

filename = '..models/model_nb1'
print("Saving model to file", filename)
pickle.dump(model, open(filename, 'wb'))

