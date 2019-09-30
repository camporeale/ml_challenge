import pandas as pd
from sklearn.model_selection import train_test_split
from spacy.lang.es import Spanish
from spacy.lang.pt import Portuguese
from spacy.tokenizer import Tokenizer
import unicodedata
from multiprocessing import Pool
from joblib import Parallel, delayed
import fasttext
import numpy as np
import csv
from projectconfig import n_cpus, data_files, normalized_files, normalized_language_files, normalized_reliable_files, models, model_files, models_for_predict
                        

def normalize_text(text,nlp):
    s = []
    for tok in nlp.tokenizer(text.lower()):
        if not tok.is_stop:
            if tok.is_alpha and not (tok.is_digit or len(tok.text) == 1):
                if not tok.is_ascii:
                    tok = ''.join(c for c in unicodedata.normalize('NFD', tok.text.lower()) if unicodedata.category(c) != 'Mn')
                    s.append(tok)
                else:
                    s.append(tok.text)
    if not s:
        return "emptystring"
    else:
        s = ' '.join(s)
        return s

    
def parallelize_dataframe(df, func):
    print("Executing parallel ", func.__name__)
    df_split = np.array_split(df, n_cpus)
    pool = Pool(n_cpus)
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
    print("Creating Train and validation files")
    # Create Train and validation files
    X_train, X_val, y_train, y_val = train_test_split(train_df["tokens"], train_df["label"], test_size=0.075, random_state=42, stratify=train_df["label"])
    train_norm = pd.concat([y_train,X_train], axis=1)
    val_norm = pd.concat([y_val,X_val], axis=1)
    
    # Saving as fasttext format
    train_norm.to_csv(csv_names["train"],index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
    val_norm.to_csv(csv_names["validation"],index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")   
    
    # Test set file
    if test_df is not None:
        print("Creating Test file")
        test_df["tokens"].to_csv(csv_names["test"],index=False,header=False,line_terminator='\n')
        test_df["language"].to_csv(normalized_language_files["mapping"]["test"],index=False,header=False,line_terminator='\n')
        

def create_language_df(train_df, test_df, csv_names):
    spanish_train = train_df[train_df["language"] == 'spanish']
    spanish_test = test_df[test_df["language"] == 'spanish']
    create_files(spanish_train, spanish_test, csv_names['spanish'])
    
    portuguese_train = train_df[train_df["language"] == 'portuguese']
    portuguese_test = test_df[test_df["language"] == 'portuguese']
    create_files(portuguese_train, portuguese_test, csv_names["portuguese"])

    
def create_reliable_df(train_df, csv_names):
    rlabel = train_df[train_df["label_quality"] == 'reliable']
    samples = rlabel.shape[0] * 1.5
    urlabel = train_df[train_df["label_quality"] == 'unreliable'].sample(n=samples, random_state=42)
    data_reliable = pd.concat([rlabel,urlabel])
    create_files(data_reliable, None, csv_names)

    
def prepare_data():
    # Load Dataset
    print("Loading datasets")
    data_train = pd.read_csv(data_files['train'])
    data_test = pd.read_csv(data_files['test'])
    print("Train and test datasets loaded")
    
    # Preprocess data
    print("Processing train dataset")
    train = parallelize_dataframe(data_train, preprocess)
    print("Processing test dataset")
    test = parallelize_dataframe(data_test, preprocess_test)

    # Create full splits
    print("Creating full csv files")
    create_files(train, test, normalized_files)

    # Create train, val and test by language
    print("Creating csv files for language specific models")
    create_language_df(train, test, normalized_language_files)

    # Create df with reliable labels oversampling
    print("Creating csv files for oversampled reliable examples")
    create_reliable_df(train, normalized_reliable_files)

    
def train_models():
    for key, value in models.items():
        print("training model:", key)
        output = fasttext.train_supervised(input=value["file"], epoch=value["epoch"], lr=value["lr"],wordNgrams=value["wordNgrams"], dim=value["dim"],thread=n_cpus)
        output.save_model(model_files[key])
        print("model", key, "saved")


def make_validation_predictions():
    for model in models:
        print("validating")
        
    
def run_model_on_test(model_file):
    test_data = pd.read_csv(normalized_files["test"],header=None,names=['title'])
    
    print("Loading model file ", model_file, '...')
    if isinstance(model_file, dict):
        # Here we predict combining models for each language
        predictions = []
        language_mapping = pd.read_csv(normalized_language_files["mapping"]["test"],names=["language"])
        test_language = pd.concat([test_data,language_mapping['language']],axis=1)
        model_sp = fasttext.load_model(model_file["spanish"])
        model_pt = fasttext.load_model(model_file["portuguese"])
        print("Running predict on test set...")
        for index, row in test_language.iterrows():
            if row["language"] == 'spanish':
                category = model_sp.predict(row["title"])[0][0]
            if row["language"] == 'portuguese':
                category = model_pt.predict(row["title"])[0][0]
            predictions.append(category[9:])
        print("Predict finished for model bilingual")
        return pd.Series(predictions, name="bilingual")
    else:            
        model = fasttext.load_model(model_file)
        print("Running predict on test set...")
        predictions = model.predict(test_data["title"].values.tolist())
        print("Predict finished for model ", model_file)
        return pd.Series([x[0][9:] for x in predictions[0]], name=model_file[10:])

def count_votes(results_df):
    voted_results = {"id": [], "category": []}
    for index, row in results_df.iloc[:,1:].iterrows():
        voted_results["id"].append(index)
        voted_results["category"].append(row.value_counts().index[0])

    voted_results_df = pd.DataFrame.from_dict(voted_results)
    return voted_results_df

def parallel_test_predict():
    results = Parallel(n_jobs=n_cpus)(delayed(run_model_on_test)(model) for name, model in models_for_predict.items())
    return results

def voting_ensemble_predict():
    results = parallel_test_predict()
    base_classifiers_results = pd.concat([x for x in results], axis=1)
    voted_results = parallelize_dataframe(base_classifiers_results, count_votes)

    return base_classifiers_results, voted_results[["id","category"]]


    
    