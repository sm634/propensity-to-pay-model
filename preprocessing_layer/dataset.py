import pandas as pd
from google.cloud import bigquery


class CornerShopData:
    """
    A Corner Shop data class. The class is intended to:
    (i) bring in corner shop data querying using BigQuery-Python interface (**SQLAlchemy)
    (ii) combine the dataset together to a single view for feature engineering.
    """

    def __init__(self):
        self.query_data = False
        self.project_id = "bold-mantis-312313"

        self.bq_app_table = self.project_id + '.curated_data_cornershop.app_usage_table_v2'
        self.bq_beacon_table = self.project_id + '.curated_data_cornershop.beacon_data_table'
        self.bq_product_table = self.project_id + '.curated_data_cornershop.product_order_table'

    def _get_app_usage(self, path="preprocessing_layer/data_temp/app_usage.csv"):
        if self.query_data:
            """
            Connect to BigQuery to access app usage table. 
            """

            client = bigquery.Client(project=self.project_id)

            query_job = client.query("""
                SELECT * FROM `""" + self.bq_app_table + """`
            """)
            app_usage = query_job.to_dataframe()

            print("Taking data from BigQuery")
            print("Rows in 'app_usage' table", app_usage.shape)

        else:
            app_usage = pd.read_csv(path)

        # Standardising data type for primary key.
        app_usage['session_id'] = app_usage['session_id'].astype(str)

        # Get unique session_ids by sorting records based on session timestamp, event date and user customer id.
        app_usage = app_usage.sort_values(by=['session_timestamp', 'event_date', 'user_customer_id'],
                                          ascending=False, na_position='last'
                                          ).drop_duplicates(subset=['session_id'])
        return app_usage

    def _get_beacon(self, path="preprocessing_layer/data_temp/beacon_data.csv"):
        if self.query_data:
            """
            Connect to BigQuery to access beacon table.
            """

            client = bigquery.Client(project=self.project_id)
            query_job = client.query("""
                SELECT * FROM `""" + self.bq_beacon_table + """`
            """)
            beacon = query_job.to_dataframe()

            print("Rows in 'beacon' table", beacon.shape)

        else:
            beacon = pd.read_csv(path)

        beacon_features_cols = ['entered_welcome', 'visited_purposeful_shelves', 'user_id', 'entered_dm',
                                'entered_store', 'visited_personalised_store', 'event_date', 'visited_digi_me',
                                'entered_purposeful_shelves']

        beacon = beacon[beacon_features_cols].copy(deep=True)
        # Get unique session_ids by sorting records based on session timestamp, event_date and user_id.
        beacon = beacon.sort_values(by=['session_timestamp', 'event_date', 'user_id'],
                                    ascending=False,
                                    na_position='last').drop_duplicates(subset=['user_id'])
        return beacon

    def _get_product(self, path="preprocessing_layer/data_temp/product_order.csv"):
        if self.query_data:
            """
            Connect to BigQuery to access product table.
            """

            client = bigquery.Client(project=self.project_id)
            query_job = client.query("""
                SELECT * FROM `""" + self.bq_product_table + """`
            """)
            product = query_job.to_dataframe()

            print("Rows in 'product' table", product.shape)

        else:
            product = pd.read_csv(path)

        # standardising session id data type
        product['session_id'] = product['session_id'].astype(str)
        # sort records to get the latest by event_date.
        product = product.sort_values(by=['event_date'], ascending=False, na_position='last')
        # product columns to use as features. Certain fields such as item_name, item_id and order_id are dropped.
        product_features_cols = ['session_id', 'price', 'quantity', 'total_cost']
        # get unique sessions for product features based at the session_id level.
        product = product[product_features_cols].groupby(by=['session_id']).sum()
        return product

    def get_single_corner_shop_data(self):
        app_usage = self._get_app_usage()
        product = self._get_product()
        beacon = self._get_beacon()

        data = app_usage.merge(product, on=['session_id'], how='left')
        data = data.merge(beacon, on=['user_id', 'event_date'], how='left')
        return data
