# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:37:38 2023
- Understand sample entropy calculation and intuition behind
- _embded is one function from utils script of antropy package
- Large values indicate high complexity, smaller values implies more self-similar and regular signals
@author: Rahul Venugopal
"""
#%% Loading libraries
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

#%% Loading custom functions
def _embed(x, order=3, delay=1):
    """Time-delay embedding.
    Parameters
    ----------
    x : array_like
        1D-array of shape (n_times) or 2D-array of shape (signal_indice, n_times)
    order : int
        Embedding dimension (order).
    delay : int
        Delay.
    Returns
    -------
    embedded : array_like
        Embedded time series, of shape (..., n_times - (order - 1) * delay, order)
    """
    x = np.asarray(x)
    N = x.shape[-1]
    assert x.ndim in [1, 2], "Only 1D or 2D arrays are currently supported."
    if order * delay > N:
        raise ValueError("Error: order * delay should be lower than x.size")
    if delay < 1:
        raise ValueError("Delay has to be at least 1.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")

    if x.ndim == 1:
        # 1D array (n_times)
        Y = np.zeros((order, N - (order - 1) * delay))
        for i in range(order):
            Y[i] = x[(i * delay) : (i * delay + Y.shape[1])]
        return Y.T
    else:
        # 2D array (signal_indice, n_times)
        Y = []
        # pre-defiend an empty list to store numpy.array (concatenate with a list is faster)
        embed_signal_length = N - (order - 1) * delay
        # define the new signal length
        indice = [[(i * delay), (i * delay + embed_signal_length)] for i in range(order)]
        # generate a list of slice indice on input signal
        for i in range(order):
            # loop with the order
            temp = x[:, indice[i][0] : indice[i][1]].reshape(-1, embed_signal_length, 1)
            # slicing the signal with the indice of each order (vectorized operation)
            Y.append(temp)
            # append the sliced signal to list
        Y = np.concatenate(Y, axis=-1)
        return Y

#%% Demo

# Set the hyper parameters for sample entropy computation
order = 3 # also called as embedding dimension
metric = "chebyshev"

# https://en.wikipedia.org/wiki/Chebyshev_distance

# 'euclidean', 'l2', 'minkowski', 'p', 'manhattan', 'cityblock', 'l1',
# 'chebyshev', 'infinity' are other distance measures available
approximate = True

# Phi is the result parameters, prealloting variable
phi = np.zeros(2)

# create some dummy data
data = np.array([1, 2, 3, 2, 3, 4, 2, 3, 4])

# visualise the data
plt.plot(data,'o-')

# r decides the criteria for closeness/error/diff x is the data
# the ddof will decide if the denominator is divided by n or n-1
# The divisor used in calculations is N - ddof
r = 0.2 * np.std(data, ddof=0) # the radius of the neighbourhood 

#%% Sample entropy calculation

# Embed the data based on order and delay
# we create sequences of three datapoints if order is 3
# if delay is 1, we shift by one datapoint and take data point 2,3,4 as the next
_emb_data1 = _embed(data, order, 1)
if approximate:
    emb_data1 = _emb_data1
else:
    emb_data1 = _emb_data1[:-1]

# Take each row (sequence) from the embed data and count how many times it
# matches with other sequences

count1 = (
    KDTree(emb_data1, metric = metric)
    .query_radius(emb_data1, r, count_only=True)
    .astype(np.float64)
)

# compute phi(order + 1, r)
emb_data2 = _embed(data, order + 1, 1)
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

# Compute the sample entropy value
-np.log(np.divide(phi[1], phi[0]))
