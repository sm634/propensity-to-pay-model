from preprocessing_layer.engineer_features import write_model_dataset
from preprocessing_layer.data_split import create_split_datasets
from preprocessing_layer.data_balance import ImbalanceCorrect
from modelling_layer.models import LogisticCLF
import pandas as pd
import argparse
from datetime import datetime


# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--bigqueryData", help="True - data import from bigquery. False - data import from csv files",
                    type=bool, required=False, default=False)
parser.add_argument("--resample", help="True if using resampling technique SMOTE to balance dataset for training",
                    type=bool, required=False, default=False)
args = parser.parse_args()

# assign argument values to variables
query_data = args.bigqueryData
resample = args.resample

# Run preprocessing layer to write model inputs to model layer
write_model_dataset(query_data)
print("corner shop created and written to csv locally.")

# split corner shop data into train and test.
create_split_datasets()
print("Data split and saved in model_inputs.")

# Apply resampling to produce a 'balanced' dataset.
if resample == 'true':
    imb_class = ImbalanceCorrect()
    balanced_train = imb_class.apply_smote()
    balanced_train.to_csv('modelling_layer/model_inputs/train_balanced.csv')
    print("balanced train data created and saved in model_inputs.")
    # train model
    logit_model = LogisticCLF(resample=True)
    logit_model.train()

else:
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

# inference on 'unseen' data - temporary data same as test set.
inference_data = pd.read_csv("modelling_layer/model_inputs/prediction_data.csv").set_index('session_id')
inference_data = inference_data.drop(labels=['completed_transaction'], axis=1)
logit_model.X = inference_data
predictions = logit_model.predict(probability=False)
pred_probs = logit_model.predict(probability=True)

pred_output = pd.DataFrame(data={"session_id": inference_data.index,
                                 "prediction": predictions,
                                 "no transaction probability": pred_probs[:, 0],
                                 "transaction probability": pred_probs[:, 1]})
dt = datetime.now()

output_path = f"modelling_layer/model_outputs/predictions_{dt}"
output_path = output_path.replace(" ", "_").replace(":", "-").replace(r".", "")
pred_output.to_csv(output_path + ".csv")
