import numpy as np
import random

import collections
from typing import List, Set, Dict, Union

import tqdm
from scipy import sparse as sp
from sklearn.cluster import KMeans
import datetime


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)


def split_comms(comms, n_train, seed, max_size=0):
    if max_size:
        comms = [x for x in comms if 3 <= len(x) <= max_size]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(comms))
    train_comms = [comms[i] for i in idx[:n_train]]
    test_comms = [comms[i] for i in idx[n_train:]]
    return train_comms, test_comms


def print_results(score, prefix='Metric'):
    p, r, f, j = score
    print(f'{prefix}: Precision:{p:.2f} Recall:{r:.2f} F1:{f:.2f} Jaccard:{j:.2f}', flush=True)


def write_comms_to_file(comms, fname):
    with open(fname, 'w') as fh:
        content = '\n'.join([' '.join([str(i) for i in comm]) for comm in comms])
        fh.write(content)


def read_comms_from_file(fname):
    with open(fname) as fh:
        content = fh.read().strip().split('\n')
        comms = [[int(i) for i in line.strip().split(' ')] for line in content]
    return comms


def get_cur_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def label_nodes(neighbors: Dict[int, Set[int]], n_roles: int) -> (np.ndarray, np.ndarray):
    """
    Compute the Jaccard similarities,
    then extract percentiles as features,
    and finally label nodes by k-means.
    """
    j_scores = collections.defaultdict(list)
    n_nodes = len(neighbors)
    for u in tqdm.tqdm(range(n_nodes), desc='ComputeJaccards'):
        u_neighbors = neighbors[u]
        for v in u_neighbors:
            if v <= u:
                continue
            v_neighbors = neighbors[v]
            js = len(u_neighbors & v_neighbors) / len(u_neighbors | v_neighbors)
            j_scores[u].append(js)
            j_scores[v].append(js)
    
    n_feats = 5
    features = np.zeros([n_nodes, n_feats])
    ps = np.linspace(0, 100, n_feats)
    for u in tqdm.tqdm(range(n_nodes), desc='ExtractPercentiles'):
        np.percentile(j_scores[u], ps, out=features[u])
    
    print("Clustering node features with KMeans...", flush=True)
    kmeans = KMeans(n_clusters=n_roles, n_init=5, algorithm='elkan', random_state=0)
    labels = kmeans.fit_predict(features)
    print("KMeans clustering done.", flush=True)
    return labels, features
