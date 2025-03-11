import source.models.fatigueset_models as FatigueSet_Models
import source.models.wesad_models as WESAD_Models


def get_embgen_model(dataset, input_dim, emb_dim):
    if dataset == 'wesad':
        embedder_model =  WESAD_Models.WESAD_Feature_Embedding_Model(input_dim, emb_dim)
    elif dataset == 'fatigueset':
        embedder_model = FatigueSet_Models.FatigueSet_Feature_Embedding_Model(input_dim, emb_dim)
    else:
        raise ValueError(f'Error: Dataset {dataset} not implemented')

    return embedder_model


def get_matching_model(dataset, emb_dim):
    matching_model = None
    if dataset == 'wesad':
        matching_model = WESAD_Models.WESAD_User_Matching_Model(emb_dim)
    elif dataset == 'fatigueset':
        matching_model = FatigueSet_Models.FatigueSet_User_Matching_Model(emb_dim)
    else:
        raise ValueError(f'Error: Dataset {dataset} not implemented')
    return matching_model

