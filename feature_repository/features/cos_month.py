import pandas as pd
import math
import numpy as np


class CosMonth:

    def __init__(self, data: pd.DataFrame):
        """
        Calculate the Cosine of the Week day data point.
        :param data: input dataset containing cornershop dataset orchestrated in the preporcessing layer.
        """
        self.data = data
        self.dependents = {'columns': [
            'session_id',
            'event_date']
        }
        self.id = 'session_id'
        self.feature_name = 'cos_wday'
        self.feature = None

    def logic(self):

        features = self.data[self.dependents['columns']].copy(deep=True)
        del self.data
        # Encoding month using a cyclical function transform month to cyclical function.
        # Normalize month to match the 0- 1/2pi cycle
        features['month'] = pd.to_datetime(features['event_date'], dayfirst=True).dt.month
        features['month_norm'] = math.pi / 2 * features['month'] / features['month'].max()
        features['cos_month'] = np.cos(features['month_norm'])

        self.feature = features[[self.id, self.feature_name]]


