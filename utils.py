import pandas as pd
from sklearn.model_selection import train_test_split
from spacy.lang.es import Spanish
from spacy.lang.pt import Portuguese
from spacy.tokenizer import Tokenizer
import unicodedata

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
    # Create Train and validation full files
    X_train, X_val, y_train, y_val = train_test_split(train["tokens"], train["label"], test_size=0.05, random_state=42, stratify=train_df["label"])
    train_norm = pd.concat([y_train,X_train], axis=1)
    val_norm = pd.concat([y_val,X_val], axis=1)
    
    # Saving as fasttext format
    train_norm.to_csv(csv_names[0],index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
    val_norm.to_csv(csv_names[1],index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")   
    print("Training: ")
    
    # Test set file
    if test_df:
        print("Processing Test file")
        test["tokens"].to_csv(csv_names[2],index=False,header=False,line_terminator='\n')
        

def create_language_files(train_df, test_df, csv_names):
    spanish_df = data_train[data_train["language"] == 'spanish']
    spanish_test = data_test[data_test["language"] == 'spanish']
    create_files(spanish_df, spanish_test, csv_names['spanish'])
        
    portuguese_df = data_train[data_train["language"] == 'portuguese']
    portuguese_test = data_test[data_test["language"] == 'portuguese']
    create_files(portuguese_df, portuguese_test, csv_names["portuguese"])
    

def create_reliable_files(train_df, csv_names):
    rlabel = train_df[data_train["label_quality"] == 'reliable']
    urlabel = data_train[data_train["label_quality"] == 'unreliable'].sample(1184245)
    data = pd.concat([rlabel,urlabel])
    create_files(portuguese_df, portuguese_test, csv_names["portuguese"])

        
def predict_test(model):
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
