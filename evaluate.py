import collections
from typing import List, Union, Set

import numpy as np


def compare_comm(pred_comm: Union[List, Set],
                 true_comm: Union[List, Set]) -> (float, float, float, float):
    """
    Compute the Precision, Recall, F1 and Jaccard similarity
    as the second argument is the ground truth community.
    """
    intersect = set(true_comm) & set(pred_comm)
    p = len(intersect) / len(pred_comm)
    r = len(intersect) / len(true_comm)
    f = 2 * p * r / (p + r + 1e-9)
    j = len(intersect) / (len(pred_comm) + len(true_comm) - len(intersect))
    return p, r, f, j


def eval_comms_bidirectional(x_comms: List[Union[List, Set]],
                             y_comms: List[Union[List, Set]]) -> (np.ndarray, np.ndarray):
    """
    Compute the P, R, F1, Jaccard scores from both two axes.
    """
    metric_names = ('precision', 'recall', 'f1', 'jaccard')
    x_node_comms = collections.defaultdict(set)
    y_node_comms = collections.defaultdict(set)
    for i, nodes in enumerate(x_comms):
        for u in nodes:
            x_node_comms[u].add(i)
    for i, nodes in enumerate(y_comms):
        for u in nodes:
            y_node_comms[u].add(i)
    
    x_neighbors = collections.defaultdict(set)
    y_neighbors = collections.defaultdict(set)
    for u in x_node_comms.keys() & y_node_comms.keys():
        x_idx = x_node_comms[u]
        y_idx = y_node_comms[u]
        for xid in x_idx:
            x_neighbors[xid].update(y_idx)
        for yid in y_idx:
            y_neighbors[yid].update(x_idx)
    cache = {}
    x_metrics = np.zeros([len(x_comms), 4])
    y_metrics = np.zeros([len(y_comms), 4])
    for i, neighbor_js in x_neighbors.items():
        np.max([cache.setdefault((i, j), compare_comm(x_comms[i], y_comms[j]))
                for j in neighbor_js], 0, out=x_metrics[i])
    for j, neighbor_is in y_neighbors.items():
        np.max([cache[(i, j)] for i in neighbor_is], 0, out=y_metrics[j])
    y_metrics[:, :2] = y_metrics[:, [1, 0]]
    if len(x_comms):
        x_mean = x_metrics.mean(axis=0)
        print('[eval] pred -> truth:')
        for name, val in zip(metric_names, x_mean):
            print(f'  {name}: {val:.6f}')
    if len(y_comms):
        y_mean = y_metrics.mean(axis=0)
        print('[eval] truth -> pred:')
        for name, val in zip(metric_names, y_mean):
            print(f'  {name}: {val:.6f}')
    return x_metrics, y_metrics
