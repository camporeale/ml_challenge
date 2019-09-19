import pandas as pd
from sklearn.model_selection import train_test_split
from spacy.lang.es import Spanish
from spacy.lang.pt import Portuguese
from spacy.tokenizer import Tokenizer
import unicodedata
from multiprocessing import  Pool
import fasttext
import numpy as np
import csv

def normalize_text(text,nlp):
    s = []
    for tok in nlp.tokenizer(text):
        if tok.is_alpha and not (tok.is_digit or tok.is_stop or len(tok.text) == 1):
            if not tok.is_ascii:
                tok = ''.join(c for c in unicodedata.normalize('NFD', tok.text.lower()) if unicodedata.category(c) != 'Mn')
                s.append(tok)
            else:
                s.append(tok.text.lower())
    if not s:
        return "emptystring"
    else:
        s = ' '.join(s)
        return s

    
def parallelize_dataframe(df, func, n_cores=8):
    print("Executing parallel ", func.__name__)
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
    return df


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
    return df


def create_files(train_df, test_df, csv_names): 
    print("Creating files")
    # Create Train and validation full files
    X_train, X_val, y_train, y_val = train_test_split(train_df["tokens"], train_df["label"], test_size=0.05, random_state=42, stratify=train_df["label"])
    train_norm = pd.concat([y_train,X_train], axis=1)
    val_norm = pd.concat([y_val,X_val], axis=1)
    
    # Saving as fasttext format
    train_norm.to_csv(csv_names["train"],index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
    val_norm.to_csv(csv_names["validation"],index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")   
    
    # Test set file
    if test_df:
        print("Processing Test file")
        test_df["tokens"].to_csv(csv_names["test"],index=False,header=False,line_terminator='\n')
        

def create_language_files(train_df, test_df, csv_names):
    spanish_train = data_train[data_train["language"] == 'spanish']
    spanish_test = data_test[data_test["language"] == 'spanish']
    create_files(spanish_train, spanish_test, csv_names['spanish'])
    
    portuguese_train = data_train[data_train["language"] == 'portuguese']
    portuguese_test = data_test[data_test["language"] == 'portuguese']
    create_files(portuguese_train, portuguese_test, csv_names["portuguese"])
    
    spanish_test["language"].to_csv('../data/language_mapping_spanish.csv', index=False, header=False)
    portuguese_test["language"].to_csv('../data/language_mapping_portuguese.csv', index=False, header=False)

    
def create_reliable_files(train_df, csv_names):
    rlabel = train_df[data_train["label_quality"] == 'reliable']
    urlabel = data_train[data_train["label_quality"] == 'unreliable'].sample(n=1184245, random_state=42)
    data = pd.concat([rlabel,urlabel])
    create_files(portuguese_df, portuguese_test, csv_names["portuguese"])

    
def prepare_data(data_files, normalized_files, normalized_language_files, normalized_reliable_files):
    print("Preparing data")
    # Load Dataset
    data_train = pd.read_csv(data_files['train'])
    data_test = pd.read_csv(data_files['test'])

    # Preprocess data
    train = parallelize_dataframe(data_train, preprocess)
    test = parallelize_dataframe(data_test, preprocess_test)

    # Create full splits
    create_files(train, test, normalized_files)

    # Create train, val and test by language
    create_language_files(train, test, normalized_language_files)

    # Create df with reliable labels oversampling
    reliable_files = ['../data/train_reliable_norm.csv','../data/val_reliable_norm.csv']
    create_reliable_df(train, reliable_files)

    
def train_models(ncpu=8):
    models = {"model_full": {"file": "../data/train_fasttext_norm.csv", "epoch": 5, "lr":0.5, "wordNgrams":2, "dim":100} ,
              "model_full_15epochs": {"file": "../data/train_fasttext_norm.csv", "epoch": 15, "lr":0.5, "wordNgrams":2, "dim":100},
              "model_full_3gram": {"file": "../data/train_fasttext_norm.csv", "epoch": 5, "lr":0.5, "wordNgrams":3, "dim":100},
              "model_reliable": {"file": "../data/train_reliable_norm.csv", "epoch": 5, "lr":0.5, "wordNgrams":2, "dim":200 },
              "model_spanish": {"file": "../data/train_spanish_norm.csv", "epoch": 5, "lr":0.5, "wordNgrams":2, "dim":200 },
              "model_portuguese": {"file": "../data/train_portuguese_norm.csv", "epoch": 5, "lr":0.5, "wordNgrams":2, "dim":200 }}
    for model in models:
        output = fasttext.train_supervised(input=model["file"], epoch=model["epoch"], lr=model["lr"],
                                           wordNgrams=model["wordNgrams"], dim=model["dim"], thread=ncpu)
        output.save_model('../models/' + model)

    
def make_ensemble_prediction(model):
    test_data = pd.read_csv(test_df,header=None,names=['title'])
    test_data['title'] = test_data['title'].apply(lambda x: ' '.join(x.split()[1:]))
    
    print("Loading model file ", model, '...')
    if isinstance(model, dict):
        # Here we predict combining models for each language
        predictions = []
        language_mapping = pd.read_csv('../data/test_language_mapping.csv',names=["language"])
        test_language = pd.concat([test_data,language_mapping['language']],axis=1)
        model_sp = fasttext.load_model(model["spanish"])
        model_pt = fasttext.load_model(model["portuguese"])
        print("Running predict on val set...")
        for index, row in test_language.iterrows():
            if row["language"] == 'spanish':
                category = model_sp.predict(row["title"])[0][0]
            if row["language"] == 'portuguese':
                category = model_pt.predict(row["title"])[0][0]
            predictions.append(category[9:])
        print("Predict finished for model ", model)
        return pd.Series(predictions)
    else:            
        model = fasttext.load_model(model)
        print("Running predict on test set...")
        predictions = model.predict(val_data["title"].values.tolist())
        print("Predict finished for model ", model)
        return pd.Series([x[0][9:] for x in predictions[0]])

    
def parallel_models_get_test_results(model_files, n_cores=5):
    results = Parallel(n_jobs=n_cores)(delayed(run_model_test)(model) for model in model_files)
    return results

def calculate_results(results):
    voted_results = {"id": [], "category": []}
    results_df = pd.concat([x for x in results], axis=1)
    for index, row in results_df.iloc[:,1:].iterrows():
        voted_results["id"].append(index)
        voted_results["category"].append(row.value_counts().index[0])

    voted_results_df = pd.DataFrame.from_dict(voted_results)
    return voted_results_df
    print("Finished")