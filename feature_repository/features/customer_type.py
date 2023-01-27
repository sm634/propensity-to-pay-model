import pandas as pd
import numpy as np

class CustomerType:

    def __init__(self, data: pd.DataFrame):
        """
        Calculate the Cosine of the Week day data point.
        :param data: input dataset containing cornershop dataset orchestrated in the preporcessing layer.
        """
        self.data = data
        self.dependents = {'columns': [
            'session_id',
            'customer_type']
        }
        self.id = 'session_id'
        self.feature_name = 'customer_type'
        self.feature = None

    def logic(self):

        features = self.data[self.dependents['columns']].copy(deep=True)
        del self.data
        features = features.replace({np.nan: 0})
        # Encoding other categorical variables numerically
        customer_type_recode = {'Returning Visitor': 0, 'New Visitor': 1}
        features.customer_type = features.customer_type.replace(customer_type_recode)

        self.feature = features[[self.id, self.feature_name]]
