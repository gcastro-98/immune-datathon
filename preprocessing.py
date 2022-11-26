"""
Implement the preprocessing steps (EDA, upsampling... also include) prior
to model training.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Any
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# customer_id: dataframe's index
onehot_features = ['education_level', 'marital_status', 'income_category',
                   'card_class']

integer_features = ['customer_age', 'number_products_customer', 'weeks_tenure',
                    'contacts_last_12mths', 'inactive_months_last_12mths',
                    'count_transactions']
float_features = ['credit_limit', 'total_revolving_balance',
                  'transactions_amount', 'change_transaction_amt_last_3mths',
                  'change_transaction_count_last_3mths']
numeric_features = integer_features + float_features

# ############################################################################
# EXPLORATORY DATA ANALYSIS
# ############################################################################


def _perform_eda(train: pd.DataFrame) -> None:
    y = train['churn']
    plt.pie(np.c_[len(y) - np.sum(y), np.sum(y)][0],
            labels=['No churn', 'Churn'], colors=['g', 'r'], shadow=True,
            autopct='%.2f')
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    plt.show()

    _compute_nan_statistics(train)


# ############################################################################
# AUXILIARY FUNCTIONS
# ############################################################################

def _reduce_mem_usage(df: pd.DataFrame, silent: bool = True,
                      _integer_features: list = None) -> pd.DataFrame:
    """
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage

    :param df: dataframe to be optimized
    :type df: pd.DataFrame
    :param silent: if False then it shows the difference of memory usage
    :type silent: bool

    :return: (pd.DataFrame) optimized dataframe
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    if not silent:
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem),
              flush=True)

    if _integer_features is None:
        _integer_features = integer_features

    for col in integer_features:
        df[col] = df[col].convert_dtypes(int)

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and \
                        c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and \
                        c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and \
                        c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and \
                        c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # We avoid using float16...
                if c_min > np.finfo(np.float32).min and \
                        c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if not silent:
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem),
              '\nDecreased by {:.1f}%'.format(
                  100 * (start_mem - end_mem) / start_mem), flush=True)

    return df


# ONEHOT ENCODING

def _generate_onehot(df: pd.DataFrame) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """
    Usage:
    :param df: dataframe containing all the features that are desired to be one-hot encoded
    :return: one-hot encoded features
    """
    # NaN values has their own category which will be also encoded
    # return pd.get_dummies(df.fillna('NaN'), prefix=df.columns)
    oh_encoder = OneHotEncoder()
    _encoded_df = oh_encoder.fit_transform(df.fillna('NaN'))
    df = pd.DataFrame(
        _encoded_df.toarray(),
        columns=oh_encoder.get_feature_names_out(df.columns),
        index=df.index)
    return df, oh_encoder


def _impute_missing_data(features: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:  # mainly for the customer age
    imp = KNNImputer(n_neighbors=10)
    _imputed_features = imp.fit_transform(features)
    if not isinstance(_imputed_features, pd.DataFrame):
        _imputed_features = pd.DataFrame(
            _imputed_features, columns=features.columns, index=features.index)
    return _imputed_features, imp


def _compute_nan_statistics(data: pd.DataFrame) -> pd.DataFrame:
    statistics = pd.DataFrame()

    # we add the percentage of samples affected by nan for each variable...
    statistics['% samples w/ nan'] = 100 * data.apply(
        lambda x: pd.isna(x)).sum(axis=0) / len(data)

    # now we generate columns indicating how likely is each variable
    # to present nan in a sample where are n nans in total,
    # for n=2,...,N_max where N_max is the number of features that
    # presents nan at least for some sample.

    n_max = np.count_nonzero(statistics > 0.)
    for n in range(2, n_max):
        more_than_one_nan = np.where(pd.isna(data), 1, 0)
        mask = (np.sum(more_than_one_nan, axis=1) - (n-1)).reshape(-1, 1)
        more_than_one_nan = np.where(more_than_one_nan * mask > 0.,
                                     more_than_one_nan, 0.)
        more_than_one_nan = 100 * pd.DataFrame(data=more_than_one_nan,
                                               columns=data.columns,
                                               index=data.index).mean(axis=0)
        statistics[f"% samples w/ {n} nan"] = more_than_one_nan

    return statistics


def __are_the_same_categories(train: pd.DataFrame, test: pd.DataFrame) -> bool:
    """
    Usage:
    _same_categories = __are_the_same_categories(
        pd.read_csv('input/train.csv', index_col=0),
        pd.read_csv('input/test.csv', index_col=0))
    :param train: raw training dataframe
    :param test: raw test dataframe
    :return: True if there are exactly the same categories for every
    categorical features in both dataframes
    """
    try:
        for col in onehot_features:
            _train_categories = pd.unique(train[col])
            _test_categories = pd.unique(test[col])
            assert set(list(_train_categories)) == set(list(_test_categories)), \
                f"There are different categories in both sets. " \
                f"Train: {_train_categories}. Test: {_test_categories} "
    except AssertionError:
        return False
    return True


def data_preprocessing() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv('input/train.csv', index_col=0)
    test = pd.read_csv('input/test.csv', index_col=0)

    _onehot_train, oh_encoder = _generate_onehot(train[onehot_features])
    train = pd.concat(
        [train.drop(onehot_features, axis=1, inplace=False),
         _onehot_train], axis=1)

    _onehot_test = pd.DataFrame(
        oh_encoder.transform(test[onehot_features].fillna('NaN')).toarray(),
        columns=oh_encoder.get_feature_names_out(onehot_features),
        index=test.index)
    test = pd.concat(
        [test.drop(onehot_features, axis=1, inplace=False),
         _onehot_test], axis=1)

    # print(_compute_nan_statistics(test))
    train_features = train[[c_ for c_ in train.columns if str(c_) != 'churn']]
    train_features, _imputer = _impute_missing_data(train_features)
    train = pd.concat([train_features, train[['churn']]], axis=1)

    test = pd.DataFrame(
        _imputer.transform(test), columns=test.columns, index=test.index)

    # print(_compute_nan_statistics(test))
    return train, test


if __name__ == '__main__':
    train, test = data_preprocessing()
    train.to_csv('input/preprocessed/train.csv', index=True)
    test.to_csv('input/preprocessed/test.csv', index=True)
