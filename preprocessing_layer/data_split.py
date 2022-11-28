import pandas as pd


def split_corner_shop_data(dataset_path='modelling_layer/model_inputs/cornershop_data.csv',
                           proportion=None,
                           val=False):
    """
    A function to split the corner shop data based on it's imbalanced nature.
    :param dataset_path:
        The corner shop dataset path to open the data from.
    :param proportion:
        A dictionary containing the proportion of train, test and val required.
    :param val:
        Whether to also create a validation set on the data or not.
    :return: a tuple of dataframes with train and test datasets (val is optional).
    """
    # Standard split of train and test.
    if proportion is None:
        proportion = {'train': 0.75, 'test': 0.25, 'val': 0}

    # read the feature engineered corners shop dataset. Currently a csv file, can be changed in future.
    dataset = pd.read_csv(dataset_path)

    # reset index for splitting.
    dataset = dataset.reset_index()

    # Split into majority and minority class based on completed transaction and sample it for splitting later.
    maj_class = dataset.loc[dataset['completed_transaction'] == 0].sample(frac=1)
    min_class = dataset.loc[dataset['completed_transaction'] == 1].sample(frac=1)

    # get the sizes of each train, test samples by class.
    maj_class_train_size = int(proportion['train'] * maj_class.shape[0])
    maj_class_test_size = int(proportion['test'] * maj_class.shape[0])

    min_class_train_size = int(proportion['train'] * min_class.shape[0])
    min_class_test_size = int(proportion['test'] * min_class.shape[0])

    # split majority and minority classes into their train and test sizes.
    train_maj = maj_class.iloc[0: maj_class_train_size]
    test_maj = maj_class.iloc[maj_class_train_size: (maj_class_train_size + maj_class_test_size)]

    train_min = min_class.iloc[0: min_class_train_size]
    test_min = min_class.iloc[min_class_train_size: (min_class_train_size + min_class_test_size)]

    # create the train and test sets.
    train = pd.concat([train_maj, train_min], axis=0)
    train = train.set_index('session_id')

    test = pd.concat([test_maj, test_min], axis=0)
    test = test.set_index('session_id')

    if val:
        maj_class_val_size = int(proportion['val'] * maj_class.shape[0])
        min_class_val_size = int(proportion['val'] * min_class.shape[0])
        val_maj = maj_class.iloc[(maj_class_train_size + maj_class_test_size):
                                 (maj_class_train_size + maj_class_test_size + maj_class_val_size)]
        val_min = min_class.iloc[(min_class_train_size + min_class_test_size):
                                 (min_class_train_size + min_class_test_size + min_class_val_size)]
        val = pd.concat([val_maj, val_min], axis=0)

        return train, test, val

    else:
        return train, test


def create_split_datasets(dataset_path='modelling_layer/model_inputs/cornershop_data.csv',
                          proportion=None,
                          val=False):
    if val:
        train, test, val = split_corner_shop_data(dataset_path,
                                                  proportion=proportion,
                                                  val=val)
        train.to_csv('modelling_layer/model_inputs/train.csv')
        test.to_csv('modelling_layer/model_inputs/test.csv')
        val.to_csv('modelling_layer/model_inputs/val.csv')
    else:
        train, test = split_corner_shop_data(dataset_path,
                                             proportion=proportion,
                                             val=val)
        train.to_csv('modelling_layer/model_inputs/train.csv')
        test.to_csv('modelling_layer/model_inputs/test.csv')
