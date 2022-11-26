from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import pdb

data = pd.read_csv("model_inputs/cornershop_data.csv").set_index('session_id')

y = data.completed_transaction.to_numpy()
X = data.drop(labels=['completed_transaction'], axis=1).to_numpy()

# train model on data
clf = LogisticRegression(random_state=42).fit(X, y)

