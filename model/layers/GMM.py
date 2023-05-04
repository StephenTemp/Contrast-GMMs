'''
--------------------------------------------------------------------------------
GMM_block imports
--------------------------------------------------------------------------------
'''
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import mahalanobis

import matplotlib.pyplot as plt
import numpy as np

'''
--------------------------------------------------------------------------------
GMM_block class
--------------------------------------------------------------------------------
'''
class GMM_block:
    model = None
    KKCs = None
    label_map = None
    threshold = None

    # initialize GMM_block
    def __init__(self, KKCs, class_map):
        self.KKCs = KKCs
        self.label_map = class_map
    
    # provided features and y, train a GMM and set
    # the mahalanobis threshold
    def train_GMM(self, X, y, verbose=True, conf=80):
        gmm = GaussianMixture(n_components=len(self.KKCs))
        y_hat = gmm.fit_predict(X, y)
        y = y.numpy().astype(int).flatten()
        self.model = gmm

        # OPTIONAL: show the learned Gaussians
        if verbose == True: 
            self.display_GMM(y_pred=y_hat, y=y, X_feats=X)
            
            gmm_acc = np.sum(y_hat == y) / len(y)
            print("Acc [Train]: ", gmm_acc)

        # get all diatances between points and corresponding 
        # cluster means
        y_dists = np.zeros(shape=y.shape)
        for i, x_i in enumerate(X):
            x_mean = gmm.means_[y[i]]
            x_cov = gmm.covariances_[y[i]]

            y_dists[i] = mahalanobis(x_i, x_mean, x_cov)

        # compute the [conf] percentile of distance distribution
        threshold = np.percentile(y_dists, q=conf)
        y_hat[y_dists > threshold] = -1
        self.threshold = threshold
        
        if verbose == True: 
            self.display_GMM(y_pred=y_hat, y=y, X_feats=X)


    def predict(self, X):
        model = self.model

        return None
    '''
    --------------------------------------------------------------------------------
        Helper Functions
    '''

    def display_GMM(self, y_pred, y, X_feats):
        y_unique = np.unique(y_pred)
        labels = self.label_map
        # means = self.model.means_

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for y_i in y_unique:
            ax.scatter(X_feats[y_pred == y_i, 0], X_feats[y_pred == y_i, 1], X_feats[y_pred == y_i, 2], label=labels[y_i], marker='o')     
            # ax.scatter(means[:, 0], means[:, 1], means[:, 2], marker='x', color='red')
        
        plt.legend()
        plt.show()
        return None