from imblearn import over_sampling
import pandas as pd


class ImbalanceCorrect:
    """
    A class that is designed to handle imbalanced datasets with off-the shelf and custom resampling techniques.
    """

    def __init__(self,
                 dataset='modelling_layer/model_inputs/train.csv'):
        """
        Extracts the dependent/target variable and features as class attributes.
        :param dataset:
            path to the training dataset so it can be balanced.
        """
        self.dataset = pd.read_csv(dataset).set_index('session_id')
        self.y = self.dataset.completed_transaction
        self.X = self.dataset.drop(labels=['completed_transaction'], axis=1)

    def apply_smote(self):
        """
        Applies off the shelf Synthetic Minority Oversampling Technique (SMOTE).
        :return:
        """
        over_sample = over_sampling.SMOTE()
        X, y = over_sample.fit_resample(self.X, self.y)
        balanced_data = pd.concat([X, y], axis=1)
        return balanced_data
