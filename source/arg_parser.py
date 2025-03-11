"""
Argument parser for the main experiments

**** IMPORTANT ****
When running experiments with BioQ, the execution order might cause variation in results
    (due to the data loader's behavior when selecting negative samples).
To minimize impact:
    1. Train the embedding generation model first: --run embgen
    2. Then train the user matching model: --run matcher
"""

import argparse
import itertools
from source.configs import DATASET_CONFIGS
SAVE_DIR='path/to/your/save/dir' # Add your save directory here


def parse_args():
    parser = argparse.ArgumentParser(description='Arg parser to run main experiments')

    # General configurations
    parser.add_argument('--config_path', type=str, default='source/data/dataset_paths.json',
                        help='Path to the configuration file containing dataset paths')
    parser.add_argument('--dataset', type=str, default='fatigueset', choices=['fatigueset', 'wesad'])
    parser.add_argument('--task', type=str, default='embgen', choices=['embgen', 'matching'],
                        help='Task to run: embgen (embedding generation) or matching (user-device matching)')
    parser.add_argument('--method', type=str, default='bioq',
                        choices=['raw', 'feature', 'lester04', 'cornelius12', 'bioq'])

    parser.add_argument('--save_dir', type=str, default=SAVE_DIR, help='Path to save results, checkpoints, etc.')
    parser.add_argument('--note', type=str, default='', help='Additional note for experiment identification')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=7, help='GPU number')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')

    # Modality, device, and window size settings
    parser.add_argument('--modals', nargs='+', default=['acc', 'gyro', 'ppg'],
                        help='List of sensor modalities to use (dataset-specific)')
    parser.add_argument('--devices', nargs='+', default=['l', 'r'], # left and right earbuds
                        help='List of devices to use (dataset-specific)')
    parser.add_argument('--ws', type=int, default=20, help='Window size in seconds (modality-specific)')


    # Evaluation settings
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train-test split ratio')
    parser.add_argument('--test_labels', type=str, default='all', help='Specify label for test (dataset-specific)')

    # Training parameters
    # # # BioQ: EmbGen
    parser.add_argument('--em_dim', type=int, default=16, help='Embedding dimension')
    parser.add_argument('--em_ep', type=int, default=400, help='Number of epochs for embedding generation')
    parser.add_argument('--em_bs', type=int, default=64, help='Batch size for embedding generation')
    parser.add_argument('--em_op', type=str, default='adam', choices=['sgd', 'adam'], help='Optimizer for embedding generation')
    parser.add_argument('--con_lr', type=float, default=0.001, help='Learning rate for contrastive learning')
    parser.add_argument('--con_wd', type=float, default=0.001, help='Weight decay for contrastive learning')

    # Samping and margin settings
    parser.add_argument('--margin', type=float, default=1.5, help='Margin for contrastive loss')
    parser.add_argument('--neg_sampling', type=str, default='middle',
                        choices=['random', 'middle', 'cut_top', 'cut_bottom'], help='Negative sampling strategy')
    parser.add_argument('--min_neg', type=int, default=10, help='Minimum negative candidates for sampling ')


    # # # BioQ: Matching
    parser.add_argument('--m_ep', type=int, default=200, help='Number of epochs for user-device matching')
    parser.add_argument('--m_bs', type=int, default=64, help='Batch size for user-device matching')
    parser.add_argument('--m_op', type=str, default='adam', choices=['sgd', 'adam'], help='Optimizer for user-device matching')
    parser.add_argument('--m_lr', type=float, default=0.001, help='Learning rate for user-device matching')
    parser.add_argument('--m_wd', type=float, default=0.001, help='Weight decay for user-device matching')

    # mnote is only used for BioQ (to identify the parameters for matching)
    parser.add_argument('--mnote', type=str, default='', help='Additional note for user-device matching')

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    """
    Validata arguments to ensure compatibility
    """
    for modal, device in itertools.product(args.modals, args.devices):
        if not DATASET_CONFIGS[args.dataset]['valid_modal_devices'][modal].__contains__(device):
            raise ValueError(f'Invalid modal({modal})-device({device}) combination in {args.dataset} dataset')

