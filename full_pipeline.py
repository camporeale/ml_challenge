import pandas as pd
from sklearn.model_selection import train_test_split
from spacy.lang.es import Spanish
from spacy.lang.pt import Portuguese
from spacy.tokenizer import Tokenizer
from sklearn.metrics import balanced_accuracy_score
from multiprocessing import  Pool
import numpy as np
import fasttext
import csv
import unicodedata

# Load Dataset
data_train = pd.read_csv('../data/train.csv')
data_test = pd.read_csv('../data/test.csv')

def normalize_text(text,nlp):
    s = []
    for tok in nlp.tokenizer(text):
        if tok.is_alpha and not (tok.is_digit or tok.is_stop or len(tok.text) == 1):
            if not tok.is_ascii:
                tok = ''.join(c for c in unicodedata.normalize('NFD', tok.text.lower()) if unicodedata.category(c) != 'Mn')
                s.append(tok)
            else:
                s.append(tok.text.lower())
    s = ' '.join(s)
    return s

def parallelize_dataframe(df, func, n_cores=8):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def preprocess(df):
    # Spacy Tokenizers
    nlp_es = Spanish()
    nlp_pt = Portuguese()
    # Spanish and Portuguese masks to use corresponding language tokenizer
    mask_spanish    = df["language"] == 'spanish'
    mask_portuguese = df["language"] == 'portuguese'
    df.loc[mask_spanish, "tokens"] = df["title"].apply(normalize_text,args=(nlp_es,))
    df.loc[mask_portuguese, "tokens"] = df["title"].apply(normalize_text,args=(nlp_pt,))
    # Training and validation df need to have __label__ string before category 
    df["label"] = df["category"].apply(lambda x: '__label__'+ x)
    return df[["label","tokens"]]

def preprocess_test(df):
    # Spacy Tokenizers
    nlp_es = Spanish()
    nlp_pt = Portuguese()
    # Spanish and Portuguese masks to use corresponding language tokenizer
    mask_spanish    = df["language"] == 'spanish'
    mask_portuguese = df["language"] == 'portuguese'
    df.loc[mask_spanish, "tokens"] = df["title"].apply(normalize_text,args=(nlp_es,))
    df.loc[mask_portuguese, "tokens"] = df["title"].apply(normalize_text,args=(nlp_pt,))
    # Test file only needs id and tokens
    return df[["id","tokens"]]

def create_fasttext_split_files(train_df, test_df):
    # Train and validation set files
    print("Processing Training and Validation splits")
    train = parallelize_dataframe(train_df, preprocess)
    X_train, X_val, y_train, y_val = train_test_split(train["tokens"], train["label"], test_size=0.05, random_state=42, stratify=train["label"])
    train_fasttext = pd.concat([y_train,X_train], axis=1)
    val_fasttext = pd.concat([y_val,X_val], axis=1)
    train_fasttext.to_csv('../data/train_fasttext_norm.csv',index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
    val_fasttext.to_csv('../data/val_fasttext_norm.csv',index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")   
    print("Training: ")
    
    # Test set file
    print("Processing Test file")
    test = parallelize_dataframe(test_df, preprocess_test)
    test["tokens"].to_csv("../data/test_fasttext_norm.csv",index=False,header=False,line_terminator='\n')


create_fasttext_split_files(data_train, data_test)

model = fasttext.train_supervised(input="../data/train_fasttext.csv", epoch=5, lr=0.5, wordNgrams=2, thread=8)

model.test('../data/val_fasttext.csv')

model.save_model("../models/model.bin")

test_data = pd.read_csv('../data/test_fasttext_norm.csv',names=['tokens'])
test_data.replace(np.nan, 'notitle',inplace=True)
predictions = model.predict(test_data["tokens"].values.tolist())
submission = pd.Series([x[0][9:] for x in predictions[0]])
submission.to_csv("./submissions/submission_test.csv",header=["category"],index_label="id")