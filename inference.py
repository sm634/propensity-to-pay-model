import pandas as pd
from modelling_layer.models import LogisticCLF
from datetime import datetime
from joblib import load

# instantiate the desired model
logit_model = load('modelling_layer/models/logisticCLF.joblib')
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
