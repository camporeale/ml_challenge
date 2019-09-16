import pandas as pd
import dask.dataframe as dd
import pyarrow

from spacy.lang.es import Spanish
from spacy.lang.pt import Portuguese
from spacy.tokenizer import Tokenizer

df = dd.read_parquet('../data/train.parquet', engine='pyarrow', npartitions=8)
print("data loaded")

#df['title'] = df['title'].str.lower()

print(df.title.value_counts().compute())

#df.to_csv('Test.csv')