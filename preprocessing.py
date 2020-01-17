from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer as BaseColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

DATE_NUMERIC = ('week',)
BUILDING_NUMERIC = ('square_feet', 'year_built')
BUILDING_CATEGORIC = ('primary_use', 'site_id')
WEATHER_NUMERIC = ('air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr',
                   'sea_level_pressure', 'wind_direction', 'wind_speed')

NUMERIC_FEATURES = DATE_NUMERIC + BUILDING_NUMERIC + WEATHER_NUMERIC
CATEGORICAL_FEATURES = BUILDING_CATEGORIC
TARGETS = ('meter_reading_0', 'meter_reading_1', 'meter_reading_2', 'meter_reading_3')
DATE_FEATURES = ('timestamp',)
READY_FEATURES = ('building_id',)


class ColumnTransformer(BaseColumnTransformer):
    """
    https://github.com/scikit-learn/scikit-learn/issues/12525#issuecomment-436217100
    """

    def get_feature_names(self):
        col_name = []
        # the last transformer is ColumnTransformer's 'remainder'
        for transformer_in_columns in self.transformers_[:-1]:
            raw_col_name = transformer_in_columns[2]
            if isinstance(transformer_in_columns[1], Pipeline):
                transformer = transformer_in_columns[1].steps[-1][1]
            else:
                transformer = transformer_in_columns[1]
            try:
                names = transformer.get_feature_names()
            except AttributeError:  # if no 'get_feature_names' function, use raw column name
                names = raw_col_name
            if isinstance(names, np.ndarray):  # eg.
                col_name += names.tolist()
            elif isinstance(names, list):
                col_name += names
            elif isinstance(names, str):
                col_name.append(names)
        return col_name


def to_datetime(string):
    return datetime.strptime(string, '%y-%m-%d %H:%M:%S')


def to_week(date_string):
    return int(to_datetime(date_string).strftime("%V"))


def preprocess(df):
    merge_columns = ['building_id', 'timestamp', 'site_id', 'primary_use', 'square_feet', 'year_built',
                     'floor_count', 'air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr',
                     'sea_level_pressure', 'wind_direction', 'wind_speed']

    df_meter_0 = get_meter_data(df, 0)
    df_meter_1 = get_meter_data(df, 1)
    df_meter_2 = get_meter_data(df, 2)
    df_meter_3 = get_meter_data(df, 3)

    df_merged_0 = df_meter_0.merge(df_meter_1, how='outer', on=merge_columns)
    df_merged_1 = df_merged_0.merge(df_meter_2, how='outer', on=merge_columns)
    df_merged = df_merged_1.merge(df_meter_3, how='outer', on=merge_columns)

    df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'])
    df_merged['week'] = df_merged['timestamp'].dt.week
    df_merged = df_merged.drop(columns=['timestamp'], axis=1)

    x_transformer = create_data_transformer(NUMERIC_FEATURES, CATEGORICAL_FEATURES, READY_FEATURES)
    x_transformer.fit(df_merged)
    X = x_transformer.transform(df_merged)

    y_transformer = create_targets_transformer(TARGETS)
    y_transformer.fit(df_merged)
    y = y_transformer.transform(df_merged)

    return X, y, x_transformer, y_transformer


def get_meter_data(df, meter):
    df_meter = df[df.meter == meter]
    df_meter = df_meter.drop(['meter'], axis=1)
    return df_meter.rename(columns={'meter_reading': f'meter_reading_{meter}'})


def create_data_transformer(
        numeric_features: Iterable[str],
        categorical_features: Iterable[str],
        ready_features: Iterable[str]) -> ColumnTransformer:
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler(feature_range=(-1, 1)))
    ])

    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, list(numeric_features)),
        ('cat', categorical_transformer, list(categorical_features)),
        ('pass', 'passthrough', list(ready_features)),
    ])


def create_targets_transformer(targets) -> ColumnTransformer:
    target_transformer = Pipeline(
        steps=[
            ('scaler', MinMaxScaler(feature_range=(0, 1)))
        ]
    )

    return ColumnTransformer(transformers=[
        ('target', target_transformer, list(targets)),
    ])
