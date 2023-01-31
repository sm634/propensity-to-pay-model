import pandas as pd
import math
import numpy as np


class CosWeekDay:

    def __init__(self, data: pd.DataFrame):
        """
        Calculate the Cosine of the Week day data point.
        :param data: input dataset containing cornershop dataset orchestrated in the preporcessing layer.
        """
        self.data = data
        self.dependents = {'columns': [
            'session_id',
            'event_wday']
        }
        self.id = 'session_id'
        self.feature_name = 'cos_wday'
        self.feature = None

    def logic(self):

        features = self.data[self.dependents['columns']].copy(deep=True)
        del self.data

        features = features.replace({np.nan: 0})

        # Encoding week days numerically
        event_wday_recode = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6,
                             'Sunday': 7}
        features['event_wday'] = features['event_wday'].replace(event_wday_recode)

        # sample cyclical encoding of weekdays.
        features['event_wday_norm'] = 1 / 2 * math.pi * features['event_wday'] / features['event_wday'].max()
        features[self.feature_name] = np.cos(features['event_wday_norm'])

        self.feature = features[[self.id, self.feature_name]]


