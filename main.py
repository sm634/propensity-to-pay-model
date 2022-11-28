from preprocessing_layer.engineer_features import write_model_dataset
from preprocessing_layer.data_split import create_split_datasets
from preprocessing_layer.data_balance import ImbalanceCorrect
from modelling_layer.models import LogisticCLF
import pandas as pd
import argparse

resample = True

# Run preprocessing layer to write model inputs to model layer
write_model_dataset()
print("corner shop created and written to csv locally.")

# split corner shop data into train and test.
create_split_datasets()
print("Data split and saved in model_inputs.")

# Apply resampling to produce a 'balanced' dataset.
if resample:
    imb_class = ImbalanceCorrect()
    balanced_train = imb_class.apply_smote()
    balanced_train.to_csv('modelling_layer/model_inputs/train_balanced.csv')
    print("balanced train data created and saved in model_inputs.")

# train model
logit_model = LogisticCLF()
logit_model.train()
# train scores
train_scores = logit_model.evaluate()
print("Train scores: \n", train_scores)

# test scores
test_data = pd.read_csv("modelling_layer/model_inputs/test.csv").set_index('session_id')
test_y = test_data.completed_transaction
test_X = test_data.drop(labels=['completed_transaction'], axis=1)
logit_model.X = test_X
logit_model.y = test_y
test_scores = logit_model.evaluate()
print("Test scores: \n", test_scores)
