import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from preprocessing_layer.dataset import CornerShopData


def get_data(query_data):
    data = CornerShopData()
    data.query_data = query_data
    data = data.get_single_corner_shop_data()
    return data


def engineer_features(query_data):
    data = get_data(query_data)
    # Getting rid of ids and columns that aren't going to be used for model effectively (e.g. transaction_value as this
    # will lead to data leakage in the case where we are predicting completed_transaction)

    # Encoding week days numerically
    event_wday_recode = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6,
                         'Sunday': 7}
    data['event_wday'] = data['event_wday'].replace(event_wday_recode)

    # Encoding other categorical variables numerically
    customer_type_recode = {'Returning Visitor': 0, 'New Visitor': 1}
    data.customer_type = data.customer_type.replace(customer_type_recode)

    # price, quantity and total_cost need to be normalised
    data = data.replace({np.nan: 0})

    minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
    data[['price', 'quantity',
          'total_cost', 'session_duration_microseconds',
          'homepage_duration_microseconds']] = minmax.fit_transform(data[['price', 'quantity',
                                                                          'total_cost', 'session_duration_microseconds',
                                                                          'homepage_duration_microseconds']])

    # Encoding month as well as weekdays using a cyclical function
    # Date and time feature engineering - only keep month from event_date and transform
    # month and time to cyclical function. Normalize month to match the 0- 1/2pi cycle
    data['month'] = pd.to_datetime(data['event_date'], dayfirst=True).dt.month
    data['month_norm'] = math.pi / 2 * data['month'] / data['month'].max()
    data['cos_month'] = np.cos(data['month_norm'])

    # sample cyclical encoding of weekdays.
    data['event_wday_norm'] = 1 / 2 * math.pi * data['event_wday'] / data['event_wday'].max()
    data['cos_wday'] = np.cos(data['event_wday_norm'])

    dataset = data[['session_id', 'customer_type', 'session_duration_microseconds', 'signed_up',
                    'pages_visited', 'app_removed', 'personalise_store_pageviews',
                    'purposeful_shelves_pageviews', 'digi_me_pageviews',
                    'homepage_duration_microseconds', 'add_to_cart', 'scan_qr',
                    'notification_open', 'notification_receive', 'price',
                    'quantity', 'total_cost', 'entered_store', 'visited_purposeful_shelves',
                    'entered_dm', 'visited_digi_me', 'visited_personalised_store',
                    'entered_welcome', 'entered_purposeful_shelves',
                    'cos_month', 'cos_wday', 'completed_transaction']].set_index('session_id')

    return dataset


def write_model_dataset(query_data, path='modelling_layer/model_inputs/cornershop_data.csv'):
    model_data = engineer_features(query_data)
    model_data.to_csv(path)
