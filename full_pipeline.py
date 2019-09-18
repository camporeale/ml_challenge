import pandas as pd
import numpy as np
import fasttext
from sklearn.metrics import balanced_accuracy_score
from multiprocessing import  Pool
from utils import preprocess, preprocess_test, parallelize

def prepare_data():
    # Load Dataset
    data_train = pd.read_csv('../data/train.csv')
    data_test = pd.read_csv('../data/test.csv')

    # Preprocess data
    train = parallelize_dataframe(data_train, preprocess)
    test = parallelize_dataframe(data_train, preprocess_test)

    # Create full splits
    full_csv_names = ['../data/train_full_norm.csv','../data/val_full_norm.csv','../data/test_full_norm.csv']
    create_files(train, test, full_csv_names)

    # Create train, val and test by language
    language_csv_names = {'spanish':['../data/train_spanish_norm.csv','../data/val_spanish_norm.csv','../data/test_spanish_norm.csv'] 
                        'portuguese': ['../data/train_portuguese_norm.csv','../data/val_portuguese_norm.csv','../data/test_portuguese_norm.csv']}
    create_language_files(train, test, language_csv_names)

    # Create df with reliable labels oversampling
    reliable_files = ['../data/train_reliable_norm.csv','../data/val_reliable_norm.csv']
    create_reliable_df(train, reliable_files)
    
def train_models():
    # Train simple model
    model1 = fasttext.train_supervised(input="../data/train_fasttext_norm.csv", epoch=5, lr=0.5, wordNgrams=2, thread=8, dim=200)
    model2 = 
    model3 = 
    model4 =
    model5 = 

    model.save_model("../models/model.bin")


def evaluate_model(model):
    model.test('../data/val_fasttext.csv')

                
def run_parallel_ensemble(model_files, n_cores=5):
    results = Parallel(n_jobs=n_cores)(delayed(run_model_test)(model) for model in model_files)
    results_df = pd.concat([x for x in results], axis=1)
    voted_results = {"id": [], "category": []}

    return results    
    
    

test_data = pd.read_csv('../data/test_fasttext_norm.csv',names=['tokens'])
test_data.replace(np.nan, 'notitle',inplace=True)
predictions = model.predict(test_data["tokens"].values.tolist())
submission = pd.Series([x[0][9:] for x in predictions[0]])
submission.to_csv("./submissions/submission_test.csv",header=["category"],index_label="id")