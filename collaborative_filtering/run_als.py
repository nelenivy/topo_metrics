import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import ParameterGrid

from common import encode_column, train_test_split, train_and_evaluate

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"

DATA_PATH = 'Movielens-20.csv'
SPLIT_QUANTILE = 0.9
PARAM_GRID = {
    'factors': [16, 32, 64, 128, 256, 512],
    'regularization': [0.01, 0.1, 1, 10],
    'alpha': [0.03, 0.1, 0.3, 1]
}
SAVE_PATH = './'
SAVE_EMBEDS = False
SAVE_RECS = False


def main():

    df = pd.read_csv(DATA_PATH)
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    print('raw data shape', df.shape)

    df, _ = encode_column(df, col='item_id', new_col='item_id')
    train, test = train_test_split(df, quantile=SPLIT_QUANTILE)

    train_csr = csr_matrix((train["rating"], (train["user_id"], train["item_id"])))

    grid_search_params = list(ParameterGrid(PARAM_GRID))
    results = []

    for params in grid_search_params:

        result, user_embeddings, item_embeddings, recs = train_and_evaluate(
            'als', train_csr, test, params)
        results.append(result)
        pd.DataFrame(results).to_csv(os.path.join(SAVE_PATH, 'ml20m_als_grid.csv'), index=False)

        filename = "_".join([f"{k}_{v}" for k, v in sorted(params.items())])
        if SAVE_EMBEDS:
            os.makedirs(os.path.join(SAVE_PATH, 'embeds/als/ml20m/users'), exist_ok=True)
            os.makedirs(os.path.join(SAVE_PATH, 'embeds/als/ml20m/items'), exist_ok=True)
            np.save(os.path.join(SAVE_PATH, 'embeds/als/ml20m/users', filename + ".npy"),
                    user_embeddings)
            np.save(os.path.join(SAVE_PATH, 'embeds/als/ml20m/items', filename + ".npy"),
                    item_embeddings)
        if SAVE_RECS:
            os.makedirs(os.path.join(SAVE_PATH, 'recs/als/ml20m'), exist_ok=True)
            recs.to_csv(os.path.join(SAVE_PATH, 'recs/als/ml20m', filename + ".csv"),
                        index=False)


if __name__ == '__main__':

    main()
