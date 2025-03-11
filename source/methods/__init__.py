
import logging
import argparse
import importlib
import pandas as pd

from source.methods.method_utils import init_wandb, create_exp_dir, load_batch_data, keep_metadata, \
    wandb_log_embgen_test_fdr, extract_and_log_test_results



def run_task(task: str, method: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
             device: str, args: argparse.Namespace, logger: logging.Logger):
    """
    :param task: 'embgen' or 'matching'
    :param method: method name
    :param train_df: training data
    :param test_df: test data
    :param device: gpu or cpu
    :param args: experiment arguments
    :param logger: logger
    """

    module_name = f"source.methods.{task}.{method.capitalize()}_{task.capitalize()}"
    # ex: methods.embgen.Bioq_Embgen
    try:
        module = importlib.import_module(module_name)
        method_class = getattr(module, f"{method.capitalize()}_{task.capitalize()}")
    except (ModuleNotFoundError, AttributeError):
        raise ValueError(f"Invalid method: {method}")

    method_instance = method_class(train_df, test_df, device, args, logger)
    method_instance.run()
