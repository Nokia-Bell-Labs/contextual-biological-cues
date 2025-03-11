import torch.nn as nn

class FatigueSet_Feature_Embedding_Model(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(FatigueSet_Feature_Embedding_Model, self).__init__()
        self.emb_dim = emb_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.projection_head = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, self.emb_dim)
        )

    def forward(self, x):
       x = self.encoder(x)
       x = self.projection_head(x)
       return x


class FatigueSet_User_Matching_Model(nn.Module):
    def __init__(self, emb_dim):
        super(FatigueSet_User_Matching_Model, self).__init__()
        self.concat_embedding_dim = emb_dim * 2 # A pair of embeddings from two devices
        self.user_matching_model = nn.Sequential(
            nn.Linear(self.concat_embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Output is a single value (Binary classification)
        )

    def forward(self, x):
        x = self.user_matching_model(x)
        return x


