from multiprocessing import  Pool
from utils import prepare_data, train_models, make_validation_predictions
from projectconfig import n_cpus, submission_file

# Preprocessing data and creating files
#prepare_data()
          
# Training all models
train_models(n_cpus)

# Make training predictions dataset
# make_validation_predictions(models)

# Train meta classifier
#train_meta_classifier()

# Make ensemble prediction
#make_ensemble_prediction()

# creating submission file
#make_submission(submission_file)