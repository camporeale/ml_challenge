import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import xgboost
from gensim.models import KeyedVectors
from sentence_embedding import EmbeddingModel, preprocess_and_compute_sentence_embedding

train = pd.read_csv('../data/train_reliable_norm.csv',header=None,names=['title'])
val = pd.read_csv('../data/val_reliable_norm.csv',header=None,names=['title'])

train['category'] = train['title'].apply(lambda x: ''.join(x.split()[0][9:]))
train['title'] = train['title'].apply(lambda x: ' '.join(x.split()[1:]))

val['category'] = val['title'].apply(lambda x: ''.join(x.split()[0][9:]))
val['title'] = val['title'].apply(lambda x: ' '.join(x.split()[1:]))

y_train = train['title']
y_val = val['category']

model = KeyedVectors.load_word2vec_format('../models/model_full_100.vec')

model = EmbeddingModel('/home/franco_camporeale/models/emb', False)

texts = train["title"].to_list()

result = preprocess_and_compute_sentence_embedding(texts[:500000],model, 'SIF', 0.1, 1)

# hay 3 sentencias que volvieron con un unico valor float 0 en vez de un array largo 300
for i, r in enumerate(result):
    if isinstance(r, float):
        result[i] = np.zeros(300)

y = y_train[:500000]

lr = LogisticRegression(multi_class='multinomial', solver='saga')

lr.fit(result, y)
