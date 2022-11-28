from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)


class LogisticCLF:

    def __init__(self, resample=False):
        """
        :param resample: boolean
            A condition to decide whether to used the balanced train data created using resampling techniques or the
            standard unbalanced dataset.
        """
        self.resample = resample
        if resample:
            data = pd.read_csv("modelling_layer/model_inputs/train_balanced.csv")
            self.clf = LogisticRegression(random_state=42, solver='liblinear', max_iter=300)
        else:
            data = pd.read_csv("modelling_layer/model_inputs/train.csv")
            data = data.set_index('session_id')
            self.clf = LogisticRegression(class_weight="balanced", random_state=42, solver='liblinear', max_iter=300)

        self.y = data.completed_transaction
        self.X = data.drop(labels=['completed_transaction'], axis=1)

    def train(self):
        # train model on data
        if self.resample:
            self.clf.fit(self.X, self.y)
        else:
            self.clf.fit(self.X, self.y)

    def evaluate(self):
        """
        Evaluate the classifier
        :return: precision, recall and f1 scores for the classifier's performance.
        """
        pred_y = self.clf.predict(self.X)
        precision = precision_score(self.y, pred_y, average="macro", zero_division=0)
        recall = recall_score(self.y, pred_y, average="macro")
        f1 = f1_score(self.y, pred_y, average="macro")

        return {"Precision: {:.4f}".format(precision),
                "Recall: {:.4f}".format(recall),
                "f1-score: {:.4f}".format(f1)}
