import numpy as np


def save_embeddings(path, compound_ids, embeddings, y_raw):
    """
    Save embeddings for visualization.
    """
    np.save(
        path,
        {
            "compound_chembl_id": compound_ids,
            "emb": embeddings,
            "Ssel": y_raw,
        },
    )
