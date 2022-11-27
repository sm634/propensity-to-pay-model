from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import pdb

# y = data.completed_transaction.to_numpy()
# X = data.drop(labels=['completed_transaction'], axis=1).to_numpy()


class LogisticCLF:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train(self):
        # train model on data
        clf = LogisticRegression(random_state=42).fit(self.X, self.y)

