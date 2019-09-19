from multiprocessing import  Pool
from utils import prepare_data, train_models

# Files provided by MercadoLibre 
data_files = {'train':'../data/train.csv', 'test': '../data/test.csv'}

# Names of files created by the preprocessing steps
normalized_files = {'train': '../data/train_full_norm.csv', 
                    'validation': '../data/val_full_norm.csv', 
                    'test': '../data/test_full_norm.csv'}

normalized_language_files = {'spanish':['../data/train_spanish_norm.csv','../data/val_spanish_norm.csv','../data/test_spanish_norm.csv'], 
                             'portuguese': ['../data/train_portuguese_norm.csv','../data/val_portuguese_norm.csv','../data/test_portuguese_norm.csv']}

normalized_reliable_files = {'train': '../data/train_reliable_norm.csv', 
                             'validation': '../data/val_reliable_norm.csv'}

submission_file = 'submission.csv'

# Preprocessing data and creating files
prepare_data(data_files,normalized_files, normalized_language_files, normalized_reliable_files)
          
# Training all models
#train_models(ncpu=8)

# Make ensemble prediction
#make_ensemble_prediction()

# creating submission file
#make_submission(submission_file)