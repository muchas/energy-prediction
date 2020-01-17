import pandas
import numpy as np

from typing import Iterable, Tuple, List


DATA_ROOT = './data'


class IntProvider:
    def get_increasing_types(self) -> Iterable[type]:
        return np.int8, np.int16, np.int32, np.int64

    def get_range(self, int_type: type) -> Tuple[int, int]:
        info = np.iinfo(int_type)
        return info.min, info.max


class FloatProvider:
    def get_increasing_types(self) -> Iterable[type]:
        return np.float16, np.float32, np.float64

    def get_range(self, float_type: type) -> Tuple[float, float]:
        info = np.finfo(float_type)
        return info.min, info.max


def reducable_type_provider(type_name: str):
    providers = (
        (IntProvider(), ('int16', 'int32', 'int64')),
        (FloatProvider(), ('float16', 'float32', 'float64'))
    )
    for provider, type_names in providers:
        if type_name in type_names:
            return provider
    return None


def reduce_column_mem_usage(df, column, type_provider):
    c_min, c_max = df[column].min(), df[column].max()
    for provided_type in type_provider.get_increasing_types():
        type_min, type_max = type_provider.get_range(provided_type)
        if c_min > type_min and c_max < type_max:
            df[column] = df[column].astype(provided_type)
            return


def reduce_memory_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for column in df.columns:
        type_provider = reducable_type_provider(df[column].dtypes)
        if not type_provider:
            continue

        reduce_column_mem_usage(df, column, type_provider)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem,
                                                                              100 * (start_mem - end_mem) / start_mem))
    return df


def read_csv(path):
    return reduce_memory_usage(pandas.read_csv(path))


def merge_datasets(meter_readings, building_metadata, weather):
    intermediate = pandas.merge(meter_readings, building_metadata, on=['building_id'], how='left')
    return pandas.merge(intermediate, weather, on=['site_id', 'timestamp'], how='left')


def train_test_split(X, y, test_size=0.2):
    train_size = int(len(X) * (1 - test_size))
    X_train, X_valid = X[:train_size], X[train_size:]
    y_train, y_valid = y[:train_size], y[train_size:]
    return X_train, X_valid, y_train, y_valid


def save_dataset(
        name: str,
        column_names: Iterable[str],
        X: np.ndarray,
        y: np.ndarray) -> None:
    np.save(f'{DATA_ROOT}/{name}_columns.npy', list(column_names))
    np.save(f'{DATA_ROOT}/{name}_data.npy', X)
    np.save(f'{DATA_ROOT}/{name}_labels.npy', y)


def load_dataset(name: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    columns = np.load(f'{DATA_ROOT}/{name}_columns.npy')
    X = np.load(f'{DATA_ROOT}/{name}_data.npy')
    y = np.load(f'{DATA_ROOT}/{name}_labels.npy')
    return columns, X, y
