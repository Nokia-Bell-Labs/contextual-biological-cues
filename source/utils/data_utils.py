from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class DataContainer:
    # 1. Actual data
    data: Dict[str, List] = field(default_factory=lambda: {
        'anchor': [], 'positive': [], 'neg_sameuser': [], 'neg_diffuser': []
    })

    # 2. User IDs
    user_id: Dict[str, List] = field(default_factory=lambda: {
        'anchor': [], 'positive': [], 'neg_sameuser': [], 'neg_diffuser': []
    })

    # 3. Device IDs
    dev_id: Dict[str, List] = field(default_factory=lambda: {
        'anchor': [], 'positive': [], 'neg_sameuser': [], 'neg_diffuser': []
    })

    # 4. Generated embeddings
    embedding: Dict[str, List] = field(default_factory=lambda: {
        'anchor': [], 'positive': [], 'neg_sameuser': [], 'neg_diffuser': []
    })

    # 5. Distances
    emb_dist: Dict[str, List] = field(default_factory=lambda: {
        'pos_dist': [], 'neg_sameuser_dist': [], 'neg_diffuser_dist': []
    })

    # Similarity based on cohesion (Lester method)
    coherence_sim: Dict[str, List] = field(default_factory=lambda: {
        'pos_sim': [], 'neg_sameuser_sim': [], 'neg_diffuser_sim': []
    })

    # 6. Timestamps
    ts: Dict[str, List] = field(default_factory=lambda: {
        'anchor': [], 'positive': [], 'neg_sameuser': [], 'neg_diffuser': []
    })

    # 7. Auxiliary task labels
    aux_label: Dict[str, List] = field(default_factory=lambda: {
        'anchor': [], 'positive': [], 'neg_sameuser': [], 'neg_diffuser': []
    })

    # 8. Session data (relevant for Fatigueset)
    session_id: Dict[str, List] = field(default_factory=lambda: {
        'anchor': [], 'positive': [], 'neg_sameuser': [], 'neg_diffuser': []
    })