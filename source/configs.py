
FatigueSet_Configs = {
    'name': 'fatigueset',
    'devices': ['l', 'r', 'h', 'w'],
    'modals': ['acc', 'gyro', 'ppg'],
    'valid_modal_devices': {
        'acc': ['l', 'r', 'h', 'w'],
        'gyro': ['l', 'r', 'h'],
        'ppg': ['l', 'r']
    }
}

WESAD_Configs = {
    'name': 'wesad',
    'devices': ['c', 'w'],
    'modals': ['acc', 'eda', 'temp'],
    'valid_modal_devices': {
        'acc': ['c', 'w'],
        'eda': ['c', 'w'],
        'temp': ['c', 'w']
    }
}

DATASET_CONFIGS = {
    'fatigueset': FatigueSet_Configs,
    'wesad': WESAD_Configs
}