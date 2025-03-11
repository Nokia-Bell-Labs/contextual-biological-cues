import logging
import argparse
import pandas as pd
from typing import Type



def load_train_test_dataframes(dataset_name: str, args: argparse.Namespace, logger: logging.Logger):
    class_path = get_class_path(args.method, 'dataset') # dataset or loader
    dataset_class = load_class_from_string(class_path)

    dataset_obj = dataset_class(dataset_name, args.config_path, args, logger)
    train_df, test_df = dataset_obj.load_train_test_data()
    return train_df, test_df


def load_windowed_dataset(dataset_name: str, df: pd.DataFrame, is_train: bool,
                          args: argparse.Namespace, logger: logging.Logger):
    """Load windowed data for the given task and dataset."""
    class_path = get_class_path(args.method, 'loader')
    loader_class = load_class_from_string(class_path)

    windowed_data_obj = loader_class(dataset_name, df, is_train, args, logger)
    return windowed_data_obj


def get_class_path(method: str, loader_type: str) -> str:
    method_map = {
        'bioq': 'feat',
        'lester04': 'raw',
        'cornelius12': 'raw',
        'raw': 'raw',
        'feature': 'feat'
    }

    data_loader_map = {
        'raw': f'raw_{loader_type}.Raw_{loader_type.capitalize()}',
        'feat': f'feat_{loader_type}.Feat_{loader_type.capitalize()}'
    }

    # Map method to data type (raw or feat)
    data_type = method_map.get(method)
    class_path = data_loader_map.get(data_type)
    full_path = f'source.data.data_loader.{data_type}.{class_path}'
    return full_path


def load_class_from_string(class_path: str) -> Type:
    module_name, class_name = class_path.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)
