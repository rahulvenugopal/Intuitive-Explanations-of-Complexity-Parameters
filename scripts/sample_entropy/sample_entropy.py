# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:07:55 2023

@author: Administrator
"""

import numpy as np
from sklearn.neighbors import KDTree

def _app_samp_entropy(x, order, metric="chebyshev", approximate=True):
    """Utility function for `app_entropy`` and `sample_entropy`."""
    _all_metrics = KDTree.valid_metrics

    phi = np.zeros(2)
    r = 0.2 * np.std(x, ddof=0)

    # compute phi(order, r)
    _emb_data1 = _embed(x, order, 1)
    if approximate:
        emb_data1 = _emb_data1
    else:
        emb_data1 = _emb_data1[:-1]
    count1 = (
        KDTree(emb_data1, metric=metric)
        .query_radius(emb_data1, r, count_only=True)
        .astype(np.float64)
    )
    # compute phi(order + 1, r)
    emb_data2 = _embed(x, order + 1, 1)
    count2 = (
        KDTree(emb_data2, metric=metric)
        .query_radius(emb_data2, r, count_only=True)
        .astype(np.float64)
    )
    if approximate:
        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
    return phi
