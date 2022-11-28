from preprocessing_layer.engineer_features import write_model_dataset
from preprocessing_layer.data_split import create_split_datasets
from preprocessing_layer.data_balance import ImbalanceCorrect
import argparse


resample = True

# Run preprocessing layer to write model inputs to model layer
write_model_dataset()
# split corner shop data into train and test.
create_split_datasets()
# Apply resampling to produce a 'balanced' dataset.
if resample:
    imb_class = ImbalanceCorrect()
    balanced_train = imb_class.apply_smote()
    balanced_train.to_csv('modelling_layer/model_inputs/train_balanced.csv')
