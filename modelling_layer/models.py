from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


class LogisticCLF:

    def __init__(self, X, y, resample=False):
        self.X = X
        self.y = y

    def train(self):
        # train model on data
        clf = LogisticRegression(random_state=42).fit(self.X, self.y)

