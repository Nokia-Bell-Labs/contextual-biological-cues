
import argparse
from source.models.model_utils import get_embgen_model, get_matching_model


def arg_parser():
    parser = argparse.ArgumentParser(description='Compute model size')
    parser.add_argument('--dataset', type=str, default='wesad', help='dataset name')
    parser.add_argument('--em_dim', type=int, default=16, help='embedding dimension')
    return parser.parse_args()


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_kb = (param_size + buffer_size) / 1024
    return size_all_kb


def main():
    args = arg_parser()
    if args.dataset == 'wesad':
        print(f"WESAD with modals acc, eda, temp")
        input_dim = 3 # acc (1) + eda (1) + temp (1) = 3
    elif args.dataset == 'fatigueset':
        print(f"FatigueSet with modals acc, gyro, ppg")
        input_dim = 5 # acc (1) + gyro (3) + ppg (1) = 5
    else:
        raise ValueError(f'Error: Dataset {args.dataset} not implemented')

    embedding_model = get_embgen_model(args.dataset, input_dim, args.em_dim)
    user_matching_model = get_matching_model(args.dataset, args.em_dim)

    print(f'Dataset: {args.dataset}')
    print(f'[1] Embedding model size: {get_model_size(embedding_model):.2f} KB')
    print(f'[2] User matching model size: {get_model_size(user_matching_model):.2f} KB')



if __name__ == '__main__':
    main()


"""
How to run?
python sys_model_size.py --dataset wesad --em_dim 16
python sys_model_size.py --dataset fatigueset --em_dim 16
"""