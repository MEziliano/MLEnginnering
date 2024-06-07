"""
This file contain all my annotation about how to implement K Nearest Neighbours - KNN from scarth only using Numpy.  

One of the first step to understand the KNN is compreend the idea of distance. 
For this script I will use euclidean distance, which is one of the most common math representation of distance.   
"""
import numpy as np 
from collections import Counter


def euclidean_distance(x1,x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance


class KNN:
    r"""
    Given a data point:
    * Calculate its distance from all other data points in the dataset
    * Get the closets K points
    ------ 
    #### Regression
        Get the average of their values
    #### Classification
        Get the label with majority vote

    """
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y 

    def predict(self, X):
        predictions = [self._predict(x)for x in X]
        return predictions
    
    def _predict(self, x):
        # compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # get the closest k 
        k_indices        = np.argsort(distances)[:self.k]
        k_nearest_lables = [self.y_train[i] for i in k_indices] 
        # majority vote 
        most_common = Counter(k_nearest_lables).most_common()
        return most_common[0][0]