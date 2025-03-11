import os
import json
import torch
import source.utils as utils
from dataclasses import asdict


def create_exp_dir(task, method, args, logger):
    assert task in ['embgen', 'matching'], "create_exp_dir // Task must be one of ['embgen', 'matching']"
    devices_str = ''.join([str(d) for d in args.devices])
    modal_str = '_'.join(sorted(args.modals))

    # Common meta
    dir1 = f'{args.save_dir}/experiments/{args.method}/{args.dataset}/' \
           f'train:{args.train_ratio}/testlbl:{args.test_labels}'
    dir2 = f'm:{modal_str}/d:{devices_str}/ws:{args.ws}/n:{args.note}/s:{args.seed}'

    if method in ['raw', 'feature', 'lester04', 'cornelius12']:
        exp_dir = f'{dir1}/{dir2}/{task}'
        os.makedirs(exp_dir, exist_ok=True)

    elif method == 'bioq':
       exp_dir, embgen_exp_dir = create_bioq_dir(task, dir1, dir2, args) # for matching, exp_dir is matching_exp_dir
       if task == 'embgen':
           os.makedirs(exp_dir, exist_ok=True)
           return exp_dir
       elif task == 'matching':
           os.makedirs(exp_dir, exist_ok=True)         # Create 'matching' directory
           os.makedirs(embgen_exp_dir, exist_ok=True)  # Create 'embgen' directory
           return exp_dir, embgen_exp_dir
       else:
           raise ValueError(f"Task {task} is not supported.")
    else:
        raise ValueError(f"Method {method} is not supported.")

    logger.info(f"Experiment directory is created as follows: \n {exp_dir}")
    return exp_dir


def create_bioq_dir(task, dir1, dir2, args):
    """Helper function to create directories for the 'bioq' method."""
    dir3 = f'edim:{args.em_dim}_eep:{args.em_ep}_ebs:{args.em_bs}_eop:{args.em_op}' \
           f'_clr:{args.con_lr}_cwd:{args.con_wd}' \
           f'_mgn:{args.margin}_ns:{args.neg_sampling}_minneg:{args.min_neg}'

    exp_dir = f'{dir1}/{dir2}/{dir3}/{task}'

    if task == 'matching':
        matching_id = f'mep:{args.m_ep}_mbs:{args.m_bs}_mop:{args.m_op}' \
                      f'_mlr:{args.m_lr}_mwd:{args.m_wd}_mn:{args.mnote}'

        matching_exp_dir = f'{dir1}/{dir2}/{dir3}/matching/{matching_id}'
        embgen_exp_dir = f'{dir1}/{dir2}/{dir3}/embgen'
        return matching_exp_dir, embgen_exp_dir
    return exp_dir, None



def init_wandb(task, method, args, logger):
    assert task in ['embgen', 'matching'], "init_wandb // Task must be one of ['embgen', 'matching']"
    try:
        import wandb
        project_name = f'BioQ_{task.capitalize()}'
        if method in ['raw', 'feature', 'lester04', 'cornelius12']:
            wandb.init(project=project_name,
                       config={'method': args.method,
                               'dataset': args.dataset,
                               'seed': args.seed,
                               'modals': args.modals,
                               'devices': args.devices,
                               'ws': args.ws,
                               'train_ratio': args.train_ratio,
                               'test_labels': args.test_labels,
                               'note': args.note})
        elif method == 'bioq':
            wandb.init(project=project_name, config=args)
        else:
            raise ValueError(f"Method {method} is not supported.")
    except Exception as e:
        logger.error(f"Error initializing wandb: {e}")



def load_batch_data(batch, win_dataset, dataset_name, train_matching=False):
    data, ts, aux_label, dev_id, user_id, session_id = {}, {}, {}, {}, {}, {}
    if train_matching:
        data_types = ['anchor', 'positive', 'negative']
    else:
        data_types = ['anchor', 'positive', 'neg_sameuser', 'neg_diffuser']

    for data_type in data_types:
        data[data_type] = batch[data_type][win_dataset.DATA_IDX]
        ts[data_type] = batch[data_type][win_dataset.WIN_IDX]
        aux_label[data_type] = batch[data_type][win_dataset.AUX_LABEL_IDX]
        dev_id[data_type] = batch[data_type][win_dataset.DEVICE_IDX]
        user_id[data_type] = batch[data_type][win_dataset.USER_IDX]
        if dataset_name == 'fatigueset':
            session_id[data_type] = batch[data_type][win_dataset.SESSION_IDX]
    # Finished looping
    if dataset_name == 'fatigueset':
        return data, (ts, aux_label, dev_id, user_id, session_id)
    return data, (ts, aux_label, dev_id, user_id)




def keep_metadata(data, metadata, dc, dataset_name):
    if dataset_name == 'fatigueset':
        ts, aux_label, dev_id, user_id, session_id = metadata
    else:
        ts, aux_label, dev_id, user_id = metadata
    for data_type in ['anchor', 'positive', 'neg_sameuser', 'neg_diffuser']:
        num_data = len(data[data_type])
        for i in range(num_data):
            dc.data[data_type].append(data[data_type][i].detach().cpu().numpy().tolist())
            dc.ts[data_type].append(ts[data_type][i].detach().cpu().numpy().tolist())
            dc.aux_label[data_type].append(aux_label[data_type][i].detach().cpu().numpy().tolist())
            dc.dev_id[data_type].append(dev_id[data_type][i].detach().cpu().numpy().tolist())
            if dataset_name == 'fatigueset':
                dc.session_id[data_type].append(session_id[data_type][i].detach().cpu().numpy().tolist())
            if torch.is_tensor(user_id[data_type][i]):
                dc.user_id[data_type].append(user_id[data_type][i].detach().cpu().numpy().tolist())
            else:
                dc.user_id[data_type].append(user_id[data_type][i])


def wandb_log_embgen_test_fdr(fdr_neg_diffuser, fdr_neg_sameuser, logger):
    try:
        import wandb
        wandb.log({'fdr_neg_diffuser': fdr_neg_diffuser, 'fdr_neg_sameuser': fdr_neg_sameuser})
    except Exception as e:
        logger.error(f"Error logging test FDR: {e}")


def extract_and_log_test_results(task, method, results, exp_dir, args, logger):
    if task == 'embgen':
        extract_and_log_test_results_embgen(method, results, exp_dir, args, logger)
    elif task == 'matching':
        extract_and_log_test_results_matching(results, exp_dir, args, logger)
    else:
        raise ValueError(f"Task {task} is not supported.")



def extract_and_log_test_results_embgen(method, dc_results, exp_dir, args, logger):
    # Save results
    result_dict = {**asdict(dc_results)}  # convert to dict
    result_dict = utils.convert_to_python_types(result_dict)

    dir_to_save = os.path.join(exp_dir, 'results')
    os.makedirs(dir_to_save, exist_ok=True)
    full_path = f'{dir_to_save}/test_distances.json'
    with open(full_path, 'w') as f:
        json.dump(result_dict, f)

    # Compute FDR (Fisher Discriminant Ratio) and save
    if method in ['bioq', 'raw', 'feature']:
        result_key_str = 'emb_dist'
        metric = 'dist'
    elif method in ['lester04', 'cornelius12']:
        result_key_str = 'coherence_sim'
        metric = 'sim'
    else:
        raise ValueError(f"Method {method} is not supported.")

    fdr_neg_diffuser = utils.fisher_discriminant_ratio(result_dict[result_key_str][f'pos_{metric}'],
                                                       result_dict[result_key_str][f'neg_diffuser_{metric}'])
    fdr_neg_sameuser = utils.fisher_discriminant_ratio(result_dict[result_key_str][f'pos_{metric}'],
                                                       result_dict[result_key_str][f'neg_sameuser_{metric}'])

    path_to_save_fdr = f'{dir_to_save}/test_fdr.json'
    with open(path_to_save_fdr, 'w') as f:
        json.dump({'fdr_neg_diffuser': fdr_neg_diffuser,
                   'fdr_neg_sameuser': fdr_neg_sameuser}, f)

    # log to wandb
    wandb_log_embgen_test_fdr(fdr_neg_diffuser, fdr_neg_sameuser, logger) if args.wandb else None

    # Print results
    print("\n\t===== EmbGen Test Results =====")
    print(f"\tModals:{args.modals}, devices: {args.devices}. ws:{args.ws}")
    print(f"\tFDR (Neg: Different Users): {fdr_neg_diffuser:.4f}"
          f"\n\tFDR (Neg: Same Unaligned Users): {fdr_neg_sameuser:.4f}")




def extract_and_log_test_results_matching(results, exp_dir, args, logger):
    result_dict = utils.convert_to_python_types(results)
    full_path = os.path.join(exp_dir, 'matching_results.json')
    with open(full_path, 'w') as f:
        json.dump(result_dict, f)

    wandb_log_matching_test_results(result_dict, logger) if args.wandb else None

    print("\n\t===== Matching Test Results =====")
    print(f"\tModals:{args.modals}, devices: {args.devices}. ws:{args.ws}")
    print(f"\tEER: {result_dict['test_eer']:.4f}, F1: {result_dict['test_f1_macro']:.4f}")


def wandb_log_matching_test_results(result_dict, logger):
    try:
        import wandb
        wandb.log({'test_eer': result_dict['test_eer'],
                   'test_f1_macro': result_dict['test_f1_macro'],
                   'test_precision': result_dict['test_precision'],
                   'test_recall': result_dict['test_recall']})
    except Exception as e:
        logger.error(f"Error logging test eer and f1 macro: {e}")

















