import pandas as pd
import numpy as np


class VisitedPersonalisedStore:

    def __init__(self, data: pd.DataFrame):
        """
        Calculate the Cosine of the Week day data point.
        :param data: input dataset containing cornershop dataset orchestrated in the preprocessing layer.
        """
        self.data = data
        self.dependents = {'columns': [
            'session_id',
            'visited_personalised_store']
        }
        self.id = 'session_id'
        self.feature_name = 'visited_personalised_store'
        self.feature = None

    def logic(self):
        features = self.data[self.dependents['columns']].copy(deep=True)
        del self.data
        features = features.replace({np.nan: 0})

        self.feature = features[[self.id, self.feature_name]]
