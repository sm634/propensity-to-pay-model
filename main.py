from preprocessing_layer.engineer_features import write_model_dataset
from preprocessing_layer.data_split import create_split_datasets


# Run preprocessing layer to write model inputs to model layer
write_model_dataset()
# split corner shop data into train and test.
create_split_datasets()
