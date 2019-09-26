from multiprocessing import  Pool
from utils import prepare_data, train_models, make_validation_predictions
from projectconfig import n_cpus, submission_file
import datetime

start = datetime.datetime.now()

# Preprocessing data and creating files
print("################","Preparing data", "################", sep="\n")
prepare_data()
          
# Training all models
print("################","Training Models", "################", sep="\n")
train_models()

# Make training predictions dataset
# make_validation_predictions(models)

# Train meta classifier
#train_meta_classifier()

# Make ensemble prediction
#make_ensemble_prediction()

# creating submission file
#make_submission(submission_file)

end = datetime.datetime.now()
print("Total time: ", end-start)