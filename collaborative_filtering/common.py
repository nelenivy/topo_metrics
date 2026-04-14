import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from replay.metrics import HitRate, MAP, NDCG, OfflineMetrics, Recall
from sklearn.preprocessing import LabelEncoder

from compute_metrics import compute_metrics


def encode_column(df, col, new_col=None, encoder=None):

    if new_col is None:
        new_col = col

    if encoder is None:
        encoder = LabelEncoder()
        df[new_col] = encoder.fit_transform(df[col])
        return df, encoder
    else:
        mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        df[new_col] = df[col].map(mapping)
        return df


def train_test_split(df, quantile=0.9):

    timeline = np.quantile(df.timestamp, quantile)

    train = df[df.timestamp < timeline]
    print('train shape', train.shape, 'users', train.user_id.nunique(),
          'items', train.item_id.nunique())

    test = df[df.timestamp >= timeline]
    test = test[test.user_id.isin(train.user_id.unique())]
    test = test[test.item_id.isin(train.item_id.unique())]
    print('test shape', test.shape, 'users', test.user_id.nunique(),
          'items', test.item_id.nunique())

    return train, test


def train_and_evaluate(model_type, train_csr, test, params, N=10, batch_size=1000):

    if model_type == 'als':
        model = AlternatingLeastSquares(iterations=15, **params)
    elif model_type == 'bpr':
        model = BayesianPersonalizedRanking(iterations=50, **params)
    else:
        raise ValueError('Unknown model type.')

    model.fit(train_csr)

    test_users = test.user_id.unique()
    recs = predict(model, test_users, train_csr, N=N, batch_size=batch_size,
                   filter_seen=True, recalculate_user=False)
    recs = recs.explode('item_id')
    recs['rating'] = recs.groupby('user_id').cumcount(ascending=False)

    offline_metrics = OfflineMetrics(
        [NDCG(N), Recall(N), HitRate(N), MAP(N)], query_column='user_id')
    rec_metrics = offline_metrics(recs, test)

    user_embeddings = model.user_factors
    user_embeddings = user_embeddings[test_users]
    item_embeddings = model.item_factors

    embed_metrics = {'user_embed_std': user_embeddings.std(),
                     'item_embed_std': item_embeddings.std(),
                     'user_embed_std_ax1': user_embeddings.std(axis=1).mean(),
                     'item_embed_std_ax1': item_embeddings.std(axis=1).mean()}

    user_metrics = compute_metrics(user_embeddings, n_samples=1, sample_fraction=1, verbose=0)
    item_metrics = compute_metrics(item_embeddings, n_samples=1, sample_fraction=1, verbose=0)

    user_metrics = {'user_' + '_'.join(k.split('_')[1:]) : v
                    for k, v in user_metrics.items() if 'metric' in k}
    item_metrics = {'item_' + '_'.join(k.split('_')[1:]) : v
                    for k, v in item_metrics.items() if 'metric' in k}

    return ({**params, **embed_metrics, **rec_metrics, **user_metrics, **item_metrics},
            user_embeddings, item_embeddings, recs)


def predict(model, user_ids, train_csr, N=10, batch_size=1000,
            filter_seen=False, recalculate_user=False):

    recs = []
    for startidx in range(0, len(user_ids), batch_size):
        batch = user_ids[startidx:startidx + batch_size]
        item_ids, scores = model.recommend(batch, train_csr[batch], N=N,
                                           filter_already_liked_items=filter_seen,
                                           recalculate_user=recalculate_user)
        recs.append(item_ids)

    recs = np.concatenate(recs)
    recs = pd.DataFrame({'user_id': user_ids, 'item_id': list(recs)})

    return recs
