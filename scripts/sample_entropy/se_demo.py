# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:37:38 2023

@author: Administrator
"""

array = np.array([[1,2,3], [2,3,4], [2,3,4]])
tree = KDTree(array, metric = 'euclidean')
neighbors = tree.query_radius(array, 1, count_only=1)

print(neighbors)
