import pandas as pd
import numpy as np
from sklearn import preprocessing


class Quantity:

    def __init__(self, data: pd.DataFrame):
        """
        Calculate the Cosine of the Week day data point.
        :param data: input dataset containing cornershop dataset orchestrated in the preprocessing layer.
        """
        self.data = data
        self.dependents = {'columns': [
            'session_id',
            'quantity']
        }
        self.id = 'session_id'
        self.feature_name = 'quantity'
        self.feature = None

    def logic(self):
        features = self.data[self.dependents['columns']].copy(deep=True)
        del self.data
        # price normalised
        features = features.replace({np.nan: 0})

        minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
        features[self.feature_name] = minmax.fit_transform(features['quantity'])

        self.feature = features[[self.id, self.feature_name]]
