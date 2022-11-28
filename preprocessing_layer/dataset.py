import pandas as pd
import sqlalchemy


class CornerShopData:
    """
    A Corner Shop data class. The class is intended to:
    (i) bring in corner shop data querying using BigQuery-Python interface (**SQLAlchemy)
    (ii) combine the dataset together to a single view for feature engineering.
    """
    def __init__(self):
        self.cred_details = ''
        self.query_data = False
        self.local_preprocess = True
        self.dataset = None

    def _get_app_usage(self, path="preprocessing_layer/data_temp/app_usage.csv"):
        if self.query_data:
            """
            BigQuery credentials to query the app usage table. 
            """
            return
        else:
            app_usage = pd.read_csv(path)

        # Standardising data type for primary key.
        app_usage['session_id'] = app_usage['session_id'].astype(str)
        # get unique session_ids.
        app_usage = app_usage.sort_values(by=['user_customer_id', 'completed_transaction'],
                                          ascending=False, na_position='last'
                                          ).drop_duplicates(subset=['session_id'])
        return app_usage

    def _get_beacon(self, path="preprocessing_layer/data_temp/beacon_data.csv"):
        if self.query_data:
            """
            BigQuery credentials to query the beacon table data. 
            """
            return
        else:
            beacon = pd.read_csv(path)

        beacon_features_cols = ['entered_welcome', 'visited_purposeful_shelves', 'user_id', 'entered_dm',
                                'entered_store', 'visited_personalised_store', 'event_date', 'visited_digi_me',
                                'entered_purposeful_shelves']
        beacon = beacon[beacon_features_cols].copy(deep=True)
        beacon = beacon.sort_values(by=['user_id', 'event_date'],
                                    ascending=False,
                                    na_position='last').drop_duplicates(subset=['user_id'])
        return beacon

    def _get_product(self, path="preprocessing_layer/data_temp/product_order.csv"):
        if self.query_data:
            """
            BigQuery credentials to query the product table data. 
            """
            return
        else:
            product = pd.read_csv(path)

        # standardising session id data type
        product['session_id'] = product['session_id'].astype(str)
        # product columns to use as features.
        product_features_cols = ['session_id', 'price', 'quantity', 'total_cost']
        #
        product = product[product_features_cols].groupby(by=['session_id']).sum()
        return product

    def get_single_corner_shop_data(self):
        app_usage = self._get_app_usage()
        product = self._get_product()
        beacon = self._get_beacon()

        data = app_usage.merge(product, on=['session_id'], how='left')
        data = data.merge(beacon, on=['user_id', 'event_date'], how='left')
        return data
