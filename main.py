"""
Main entry point for the project.
Refer to arg_parser.py for the list of arguments and their default values.
"""

import time
import torch
import logging
from source.utils import set_seed
from source.methods import run_task
from source.arg_parser import parse_args
from source.data.data_loader import load_train_test_dataframes

# Logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s',datefmt='%Y-%m-%d %H:%M:%S')


def main():
    st = time.time()
    args = parse_args()

    # Set logger, seed and device
    logger = logging.getLogger()
    set_seed(args.seed)
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    # Load data
    try:
        train_df, test_df = load_train_test_dataframes(args.dataset, args, logger)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    logger.info(f"[{args.task}] Running {args.method} on {args.dataset}...")

    # Run task
    try:
        run_task(args.task, args.method, train_df, test_df, device, args, logger)
    except Exception as e:
        logging.error(f"Error running task: {e}")
        return
    logger.info(f"Total time taken: {time.time()-st:.3f} seconds")

if __name__ == '__main__':
    main()