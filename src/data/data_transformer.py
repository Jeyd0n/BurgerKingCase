import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.base import BaseEstimator, TransformerMixin


class DataTransformer(BaseEstimator, TransformerMixin):
    """
    Данный класс используется для преобразования сырого набора данных в датасет,
    используемый при тренировке и в инференсе
    """

    def __init__(self, is_train=True):
        self.is_train = is_train

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X.drop('group_name', axis=1, inplace=True)

        X['MaxMinDelta'] = (
            X.groupby('customer_id')['startdatetime'].transform('max') - X.groupby('customer_id')['startdatetime'].transform('min')
        ).dt.days

        if self.is_train == True:
            X['date_diff_post'].fillna(
                value=0,
                inplace=True
            )

        unique_formats = X['format_name'].unique()
        format_dict = {
            value: idx  for idx, value in enumerate(unique_formats)
        }
        X['format_name'] = X['format_name'].replace(format_dict)

        unique_dishes = X['dish_name'].unique()
        dishes_dict = {
            value: idx  for idx, value in enumerate(unique_dishes)
        }
        X['dish_name'] = X['dish_name'].replace(dishes_dict)

        X['OrderPrice'] = X.groupby(['customer_id', 'startdatetime'])['revenue'].transform('sum')
        X['MeanOrderPrice'] = round(X.groupby('customer_id')['OrderPrice'].transform('mean'), 2)
        X.drop('revenue', axis=1, inplace=True)

        X['FavoriteDish'] = X.groupby('customer_id')['dish_name'].transform(lambda x: x.value_counts().index[0])
        X.drop('dish_name', axis=1, inplace=True)

        X['hour'] = X['startdatetime'].dt.hour
        X['minute'] = X['startdatetime'].dt.minute
        X['FavoriteHour'] = X.groupby('customer_id')['hour'].transform(lambda x: x.value_counts().index[0])

        X['MostVisitedFormat'] = X.groupby('customer_id')['format_name'].transform(lambda x: x.value_counts().index[0])

        X.drop(['format_name', 'OrderPrice', 'hour', 'minute', 'ownareaall_sqm'], axis=1, inplace=True)
        X.drop_duplicates(
            subset='customer_id',
            inplace=True
        )

        return X