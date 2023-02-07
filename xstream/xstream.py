#!/usr/bin/env python3

import numpy as np
import random
import mmh3
import itertools
import numpy as np
from sklearn.metrics import roc_auc_score

class StreamhashProjection:

    def __init__(self, n_components, density=1/3.0, random_state=None):
        self.keys = np.arange(0, n_components, 1)
        self.constant = np.sqrt(1./density)/np.sqrt(n_components)
        self.density = density
        self.n_components = n_components
        random.seed(random_state)

    def fit_transform(self, X, feature_names=None):
        nsamples = X.shape[0]
        ndim = X.shape[1]
        if feature_names is None:
            feature_names = [str(i) for i in range(ndim)]

        R = np.array([[self._hash_string(k, f)
                       for f in feature_names]
                       for k in self.keys])

        # check if density matches
        #print "R", np.sum(R.ravel() == 0)/float(len(R.ravel()))

        Y = np.dot(X, R.T)
        return Y

    def transform(self, X, feature_names=None):
        return self.fit_transform(X, feature_names)
    
    def _hash_string(self, k, s):
        hash_value = int(mmh3.hash(s, signed=False, seed=k))/(2.0**32-1)
        s = self.density
        if hash_value <= s/2.0:
            return -1 * self.constant
        elif hash_value <= s:
            return self.constant
        else:
            return 0
        
        


class Chain:

    def __init__(self, deltamax, depth=25):
        k = len(deltamax)
        self.deltamax = deltamax # feature ranges
        self.depth = depth
        self.fs = [np.random.randint(0, k) for d in range(depth)]
        self.cmsketches = [None] * depth
        self.shift = np.random.rand(k) * deltamax

    def fit(self, X, verbose=False, update=False):
        prebins = np.zeros(X.shape, dtype=np.float)
        depthcount = np.zeros(len(self.deltamax), dtype=np.int)
        for depth in range(self.depth):
            f = self.fs[depth]
            depthcount[f] += 1

            if depthcount[f] == 1:
                prebins[:,f] = (X[:,f] + self.shift[f])/self.deltamax[f]
            else:
                prebins[:,f] = 2.0*prebins[:,f] - self.shift[f]/self.deltamax[f]

            if update:
                cmsketch = self.cmsketches[depth]
            else:
                cmsketch = {}
            for prebin in prebins:
                l = tuple(np.floor(prebin).astype(np.int))
                if not l in cmsketch:
                    cmsketch[l] = 0
                cmsketch[l] += 1
            self.cmsketches[depth] = cmsketch
        return self

    def bincount(self, X):
        scores = np.zeros((X.shape[0], self.depth))
        prebins = np.zeros(X.shape, dtype=np.float)
        depthcount = np.zeros(len(self.deltamax), dtype=np.int)
        for depth in range(self.depth):
            f = self.fs[depth] 
            depthcount[f] += 1

            if depthcount[f] == 1:
                prebins[:,f] = (X[:,f] + self.shift[f])/self.deltamax[f]
            else:
                prebins[:,f] = 2.0*prebins[:,f] - self.shift[f]/self.deltamax[f]

            cmsketch = self.cmsketches[depth]
            for i, prebin in enumerate(prebins):
                l = tuple(np.floor(prebin).astype(np.int))
                if not l in cmsketch:
                    scores[i,depth] = 0.0
                else:
                    scores[i,depth] = cmsketch[l]

        return scores

    def score(self, X, adjusted=False):
        # scale score logarithmically to avoid overflow:
        #    score = min_d [ log2(bincount x 2^d) = log2(bincount) + d ]
        scores = self.bincount(X)
        depths = np.array([d for d in range(1, self.depth+1)])
        scores = np.log2(1.0 + scores) + depths # add 1 to avoid log(0)
        return np.min(scores, axis=1)

class Chains:
    def __init__(self, k=50, nchains=100, depth=25, seed=42):
        self.nchains = nchains
        self.depth = depth
        self.chains = []
        self.projector = StreamhashProjection(n_components=k,
                                              density=1/3.0,
                                              random_state=seed)

    def fit(self, X):
        projected_X = self.projector.fit_transform(X)
        deltamax = np.ptp(projected_X, axis=0)/2.0
        deltamax[deltamax==0] = 1.0
        for i in range(self.nchains):
            c = Chain(deltamax, depth=self.depth)
            c.fit(projected_X)
            self.chains.append(c)

    def score(self, X, adjusted=False):
        projected_X = self.projector.transform(X)
        scores = np.zeros(X.shape[0])
        for i in (range(self.nchains)):
            chain = self.chains[i]
            scores += chain.score(projected_X, adjusted)
        scores /= float(self.nchains)
        return scores
        
        
class XStream():
    def __init__(self,k=50,nchains=50,depth=10, **kwargs):
        # initialize
        self.xStream = None
        self.k = k
        self.nchains = nchains
        self.depth = depth
        self.initialize_xStream(**kwargs)


    def initialize_xStream(self, seed=0, **kwargs):
        self.xStream = Chains(k=self.k, nchains=self.nchains, depth=self.depth, **kwargs)

    def fit(self, train_X):
        print("Starting training...")
        self.xStream.fit(train_X.astype(np.float32))

    def predict_proba(self,test_X):
        scores = (-1.0) * self.xStream.score(test_X.astype(np.float32))  # compute anomaly score
        return scores
    
    def predict(self,test_X,test_y):
        print("Starting prediction...")
        scores = (-1.0) * self.xStream.score(test_X.astype(np.float32))  # compute anomaly score
        auc = roc_auc_score(test_y, scores.flatten())
        print("AUCROC: %.4f" % auc)
        return scores