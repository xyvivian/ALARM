#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Xstream: adapted to categorical/ mixed-type data
Reference: @inproceedings{10.1145/3219819.3220107,
           author = {Manzoor, Emaad and Lamba, Hemank and Akoglu, Leman},
           title = {XStream: Outlier Detection in Feature-Evolving Data Streams},
           year = {2018},
           isbn = {9781450355520},
           publisher = {Association for Computing Machinery},
           url = {https://doi.org/10.1145/3219819.3220107},
           series = {KDD '18}
           }
          
           @inproceedings{10.1145/3534678.3539076,
            author = {Zhang, Sean and Ursekar, Varun and Akoglu, Leman},
            title = {Sparx: Distributed Outlier Detection at Scale},
            year = {2022},
            isbn = {9781450393850},
            publisher = {Association for Computing Machinery},
            url = {https://doi.org/10.1145/3534678.3539076},
            series = {KDD '22}
            }
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import IsolationForest
import sys
import pandas as pd
from copy import deepcopy
from numpy import linalg as LA
import json

from sklearn.manifold import MDS
from matplotlib import pyplot as plt
import sklearn.datasets as dt
import seaborn as sns         
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import math
import random
import mmh3
import tqdm
import sys
sys.path.append("..")




def get_mds(shap_inference):
    """
    Get 2 dim projection
    :param shap_inference: shap inference array
    :return: dataframe with x and y values of mds
    """
    dist_euclid = euclidean_distances(shap_inference)
    mds = MDS(dissimilarity="precomputed", random_state=0)
    data_transformed = mds.fit_transform(dist_euclid)
    data_MDS = pd.DataFrame(data_transformed, columns=["x", "y"])
    return data_transformed, data_MDS


def _hash_string(k, s):
    hash_value = int(mmh3.hash(s, signed=False, seed=k))/(2.0**32-1)   
    den = 1/3
    if hash_value <= den/2.0:
        return 1 #-1 
    elif hash_value <= den:
        return 1
    else:
        return 0
    

class StreamhashProjection:

    def __init__(self, n_components, density=1/3.0, random_state=None):
        self.keys = np.arange(0, n_components, 1)
        self.constant = np.sqrt(1./density)/np.sqrt(n_components)
        self.density = density
        self.n_components = n_components
        random.seed(random_state)
        self.is_R = False
        self.feature_names = None
        self.R = None
        
    def initialize_R(self, X,feature_names):
        ndim = X.shape[0]
        if feature_names is None:
            feature_names = [str(i) for i in range(ndim)]
        types = [type(X[i]) == str for i in range(ndim)]
        feature_name = []
        for i in range(ndim):
            if types[i]:
                feature_name.append("%s%s%s" %(feature_names[i],'.',X[i]))
            else:
                feature_name.append(feature_names[i])
        feature_names = feature_name        
        self.R = np.array([[_hash_string(k, f)
                       for f in feature_names]
                       for k in self.keys])
        for i in range(ndim):
            if types[i]:
                f = feature_names[i]
                self.R[:,i] = np.array([_hash_string(k, f) for k in self.keys])
        self.is_R = True
        self.feature_names =feature_names 
        

    def fit_transform(self, X, feature_names=None): 
        if not self.is_R:
            self.initialize_R(X,feature_names)
            
        ndim = X.shape[0] 
        types = [type(X[i]) == str for i in range(ndim)]
        X = [1 if types[i] else X[i] for i in range(ndim)]
        Y = np.dot(X, self.R.T)       
        return Y

    def transform(self, X, feature_names=None):
        return self.fit_transform(X, feature_names)
    
    


class Chain:

    def __init__(self, deltamax, depth=25):
        k = len(deltamax)
        self.deltamax = deltamax # feature ranges
        self.depth = depth
        self.fs = [np.random.randint(0, k) for d in range(depth)]
        self.cmsketches = [None] * depth
        self.shift = np.random.rand(k) * deltamax

    def fit(self, X, verbose=False, update=False):
        prebins = np.zeros(X.shape, dtype=float)
        depthcount = np.zeros(len(self.deltamax), dtype=int)
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
                l = tuple(np.floor(prebin).astype(int))
                if not l in cmsketch:
                    cmsketch[l] = 0
                cmsketch[l] += 1
            self.cmsketches[depth] = cmsketch
        return self

    def bincount(self, X):
        scores = np.zeros((X.shape[0], self.depth))
        prebins = np.zeros(X.shape, dtype=float)
        depthcount = np.zeros(len(self.deltamax), dtype=int)
        for depth in range(self.depth):
            f = self.fs[depth] 
            depthcount[f] += 1

            if depthcount[f] == 1:
                prebins[:,f] = (X[:,f] + self.shift[f])/self.deltamax[f]
            else:
                prebins[:,f] = 2.0*prebins[:,f] - self.shift[f]/self.deltamax[f]

            cmsketch = self.cmsketches[depth]
            for i, prebin in enumerate(prebins):
                l = tuple(np.floor(prebin).astype(int))
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
    
    def score_all_depths(self,X):
        scores = self.bincount(X)
        depths = np.array([d for d in range(1, self.depth+1)])
        scores = -(np.log2(1.0 + scores) + depths)
        return scores


class Chains:
    def __init__(self, k=50, nchains=100, depth=25, seed=42, projection = False):
        self.nchains = nchains
        self.depth = depth
        self.chains = []
        self.is_projection = projection
        if self.is_projection:
            self.projector = StreamhashProjection(n_components=k,
                                              density=1/3.0,
                                              random_state=seed)
    def get_projection(self,X,feature_names=None):
        if not self.is_projection:
            return X
        else:
            projected_X = []
            for i in range(X.shape[0]):
                if type(X) == np.ndarray:
                    val = X[i]
                else:
                    val = X.iloc[i]
                projected_X.append(self.projector.fit_transform(val,feature_names))
            return np.array(projected_X)
    
    def fit(self, projected_X):
        deltamax = np.ptp(projected_X, axis=0)/2.0
        deltamax[deltamax==0] = 1.0
        for i in tqdm.tqdm(range(self.nchains), desc='Fitting...'):
            c = Chain(deltamax, depth=self.depth)
            c.fit(projected_X)
            self.chains.append(c)

    def score(self, projected_X, adjusted=False):
        scores = np.zeros(projected_X.shape[0])
        for i in tqdm.tqdm(range(self.nchains), desc='Scoring...'):
            chain = self.chains[i]
            scores += chain.score(projected_X, adjusted)
        scores /= float(self.nchains)
        return scores


def score_in_chains(cf, X_to_explain):
    """
    For each outlier point in topk datapoints,
    we detect the ch_chains that contribute to the outlier points
    and take the average as the feature_importance
    
    If projection is set to true, the output is fed into Expalantion class
    to trace back the feature importances in the original feature space
    
    Input: cf: Xstream chains
           X_to_explain: the data points to explain
           
    Ooiutput: np.ndarray, fimportance
    """
    fused = np.zeros((X_to_explain.shape[0],X_to_explain.shape[1], cf.nchains))
    score_in_c = np.zeros(X_to_explain.shape)
    #max chain val
    allscores = []
    for cindex in range(0,cf.nchains):
        c = cf.chains[cindex]
        score_c_AllDepths = -c.score_all_depths(X_to_explain)
        allscores.append(score_c_AllDepths)
    
    max_split_val = np.max(np.array(allscores))
    #calculate the scores on splits of each chains
    for cindex in range(0, cf.nchains):
        c = cf.chains[cindex]
        score_c_AllDepths = -c.score_all_depths(X_to_explain)

        ind_min = np.argmin(score_c_AllDepths,axis = 1)
        score_c = np.zeros((X_to_explain.shape[0],))
        for idx,ind in enumerate(ind_min):
            score_c[idx] = score_c_AllDepths[idx,ind]
        cSplitFeatures = c.fs
        
        for idx,ind in enumerate(ind_min):
            for split_ind, split_f in enumerate(cSplitFeatures[0:ind+1]):
                fused[idx,split_f,cindex] = 1
                score_in_c[idx,split_f] = score_in_c[idx,split_f] + (max_split_val - score_c_AllDepths[idx,split_ind])
                    
    fimportance = np.zeros(X_to_explain.shape)
    for f in range(fimportance.shape[1]):
        #instead of assigining it to zero, we assign +1 to all values
        num_in = np.sum(fused[:,f,:],axis =1) + 1.0
        fimportance[:,f] = score_in_c[:,f]/num_in

    print("get Importance of (projection) features BY SCORE ")
    return(fimportance)
  

class Explanation:
    def __init__(self,X,feature_names, explanation_type, projdim):
        self.ndim = len(feature_names)
        self.X = X
        self.projdim = projdim
        self.explanation_type = explanation_type
        self.feature_names = feature_names
        if type(X) == np.ndarray:
            self.types = [type(X[0][i]) == str for i in range(self.ndim)]
        else:
            self.types = [type(X.iloc[0][feat]) == str for feat in feature_names]
        #self.R_ohe = self.get_Rohe(X)

        
    def explain(self,fimportance,X):
        explain = []
        for i in range(fimportance.shape[0]):
            if self.explanation_type == "average":
                explain.append(self.avg_projection(fimportance[i,:],X[i,:]))
            elif self.explanation_type == "random_walk":
                explain.append(self.random_walk_projection(fimportance[i,:],X[i,:]))
            elif self.explanation_type == "hits":
                explain.append(self.hits_projection(fimportance[i,:],X[i,:]))
            elif self.explanation_type == "signed_hits":
                explain.append(self.signed_hits_projection(fimportance[i,:],X[i,:]))
            elif self.explanation_type == "random_walk2":
                explain.append(self.random_walk_projection2(fimportance[i,:],X[i,:]))
        #now explain should equal to topk times total_feature_dimensions
        #if we have one-hot-encoding, we should only select the feature index with "1".
        return explain

    def get_Rohe(self,X):
        ndim = X.shape[0]
        if self.feature_names is None:
            self.feature_names = [str(i) for i in range(ndim)]
        types = [type(X[i]) == str for i in range(ndim)]
        feature_name = []
        for i in range(ndim):
            if types[i]:
                feature_name.append("%s%s%s" %(self.feature_names[i],'.',X[i]))
            else:
                feature_name.append(self.feature_names[i])
        feature_names = feature_name 
        keys = np.arange(0,self.projdim,1)
        ohe_R = np.array([[_hash_string(k, f)
                       for f in self.feature_names]
                       for k in keys])
        for i in range(ndim):
            if types[i]:
                f = feature_names[i]
                ohe_R[:,i] = np.array([_hash_string(k, f) for k in keys])
        return ohe_R


    def avg_projection(self,fimportance,X):   
        #get the ohe_to the X
        R_ohe = self.get_Rohe(X)
        fimpr = deepcopy(fimportance) # which one to use
        totalprojperfeat = np.sum(R_ohe, axis=0)
        #print(totalprojperfeat)
        #totalfeatureperproj = np.sum(self.R_ohe, axis=1)
        scaledR = R_ohe.copy()
        for p in range(R_ohe.shape[0]):
            scaledR[p,:] = scaledR[p,:] * fimpr[p]
        origfimportance = np.sum(scaledR, axis=0) / totalprojperfeat
        return origfimportance
    
    
    def random_walk_projection2(self,fimportance,X):
        R_ohe = self.get_Rohe(X)
        pagerank = PageRank(damping_factor=0.2,solver ='diteration')
        adjacency = R_ohe
        seeds = fimportance.reshape((R_ohe.shape[0],)) # / sum(fimportance.reshape((15,)))
        seeds = seeds-np.min(seeds)
        pagerank.fit(adj, seeds_row = seeds,force_bipartite = True)
        return pagerank.scores_col_


    def random_walk_projection(self,fimportance,X,alpha = 0.5,repeat_times = 10):
        # attribution to original features
        # 1. BY RANDOM WALKS
        R_ohe = self.get_Rohe(X)
        dr = np.random.rand(R_ohe.shape[1],1)
        pr = np.random.rand(R_ohe.shape[0],1)
        totalprojperfeat = np.sum(R_ohe, axis=0)
        sumr = dr.sum()+pr.sum()
        dr = dr / sumr
        pr = pr / sumr
        fimpr = fimportance.copy()
        newR = R_ohe.copy()
        for p in range(R_ohe.shape[0]):
            newR[p,:] = newR[p,:] * fimpr[p]
        origfimportance = np.sum(newR, axis=0) / totalprojperfeat
        origfimportance = np.nan_to_num(origfimportance)
        #Option 1: normalize the features
        if np.min(fimpr) < 0:
            fimpr = fimpr + np.abs(np.min(fimpr))  
        fimpr = fimpr / sum(fimpr) # normalize so sum to 1, a prob dist.n
        #print(fimpr)
        fimpr = fimpr.reshape(newR.shape[0],1)
        sums = newR.sum(axis=0,keepdims=1)
        sums[sums==0] = 1
        A = newR.copy() / sums
        B = newR.copy().T
        sums = B.sum(axis=0,keepdims=1)
        sums[sums==0] = 1
        B = B / sums
        for i in range(repeat_times):
            #print(i)
            drnew = alpha * np.dot(B, pr)
            #print(drnew.shape)
            prnew = alpha * np.dot(A, dr) + (1-alpha) *fimpr 
            #print(prnew.shape)
            denom = (drnew.sum()+prnew.sum())
            dr = drnew.reshape(R_ohe.shape[1],1) / denom
            pr = prnew.reshape(R_ohe.shape[0],1) / denom
        return origfimportance

    def hits_projection(self,fimportance,X,alpha=0.5,repeat_time = 20):
        R_ohe = self.get_Rohe(X)
        newR = R_ohe.copy()
        fimpr = fimportance.copy() 
        # Shift so that minimum is zero
        if np.min(fimpr) < 0:
            fimpr = fimpr + np.abs(np.min(fimpr))

        fimpr = fimpr / LA.norm(fimpr,2)
        fimpr = fimpr.reshape(R_ohe.shape[0],1)
        pr = fimpr.copy()
        B = newR.T
        sums = B.sum(axis=0,keepdims=1)
        sums[sums==0] = 1
        B = B / sums
        for i in range(repeat_time):
            dr = np.dot(B, pr)
            dr = dr.reshape(newR.shape[1],1) / LA.norm(dr,2)
            pr = np.dot(newR, dr)
            pr = alpha * pr + (1-alpha) * fimpr
            pr = pr.reshape(newR.shape[0],1) / LA.norm(pr,2)   
        return dr

    def signed_hits_projection(self,fimportance,X, epsilon = 0.1, repeat_time = 20):
        R_ohe = self.get_Rohe(X)
        dr = np.ones(R_ohe.shape[1]) * epsilon # importance
        pr = np.ones(R_ohe.shape[0]) * epsilon # authority/accurateness
        B = R_ohe.copy().T
        sums = B.sum(axis=0,keepdims=1)
        sums[sums==0] = 1
        scaledR = R_ohe.copy()
        scaledRtrans = scaledR.T / sums
        for i in range(repeat_time):
            dr = np.dot(scaledRtrans, pr)
            dr = dr.reshape(scaledR.shape[1],1) / LA.norm(dr,2) # abs(dr.sum())
            pr = np.dot(scaledRtrans.T, dr)
            pr = pr.reshape(scaledR.shape[0],1) / LA.norm(pr,2) # abs(pr.sum())
        return dr


class Parameters():
    def __init__(self, 
                 json_file_name,
                 ):
        with open(json_file_name, 'r') as openfile:
            json_ = json.load(openfile)
        self.projection = json_["projection"]
        self.input_file = json_["input_file"]
        self.projdim = json_["projdim"]
        self.nchains = json_["nchains"]
        self.depth = json_["depth"]
        self.output_path = json_["output_path"]
        self.dataset_name = json_["dataset_name"]
        self.explain = json_["explain"]
        self.explain_method = json_["explain_method"]
        self.topk = json_["topk"]
        self.cluster_num = json_["cluster_num"]
        self.has_label = json_["has_label"]
        self.use_label = json_["use_label"]
        
        
def main(file_path = "xstream_parameters.json"):
    parameters = Parameters(file_path)
    projdim = parameters.projdim
    nchains = parameters.nchains
    depth = parameters.depth
    projection = parameters.projection
    input_file = parameters.input_file
    explain_type = parameters.explain_method
    output_path = parameters.output_path
    is_explain = parameters.explain
    explain_method = parameters.explain_method
    has_label = parameters.has_label
    use_label = parameters.use_label
    #create output directory
    isExist = os.path.exists(output_path)
    if not isExist:
        os.makedirs(output_path)  
    
    #load the data 
    data = pd.read_csv(input_file,delimiter=",",index_col='index')
    
    list_cat_name = []
    list_cat = []
    list_flt = []
    for i in range(len(data.dtypes)):
        if data.dtypes[i] == int or data.dtypes[i] == float:
            list_flt.append(i)
        else:
            list_cat.append(i)
            list_cat_name.append(data.columns[i])
    
    if has_label:
        feature_names = list(data.columns)[0:-1]
        label_name = list(data.columns)[-1]
        Y = data[label_name]
        X = data[feature_names]
    else:
        feature_names = list(data.columns)
        X = data[feature_names]
        
        
    topk = parameters.topk
    if has_label == True and use_label == True:
        anomaly_index = [idx for idx,i in enumerate(Y) if i == 1]
        topk = len(anomaly_index)
    
    val = X.to_numpy()
    
    cf = Chains(k=projdim, nchains=nchains, depth=depth, projection = projection)
    projected_X = cf.get_projection(val,feature_names)
    cf.fit(projected_X)
    anomalyscores = -cf.score(projected_X)
    if has_label:
        ap = average_precision_score(Y, anomalyscores) 
        auc = roc_auc_score(Y, anomalyscores)
        print("xstream: AP =", ap, "AUC =", auc)
    
    anomalyscores = (anomalyscores - anomalyscores.min(axis=0)) / (anomalyscores.max(axis=0) - anomalyscores.min(axis=0))
    
    # find the top k
    if use_label is True:
        if has_label is False:
            print("Should provide actual labels, exit the program")
            exit()
        else:
            top_index = np.array( [idx for idx,i in enumerate(Y) if i == 1])
    else:
        top_index = np.argsort(anomalyscores)[::-1][0:topk]
    X_explain = projected_X[top_index]
    if has_label:
        result = np.concatenate((top_index.reshape(topk,1), anomalyscores[top_index].reshape(topk,1),\
                         Y.to_numpy()[top_index].reshape(topk,1)), axis = 1)
        print("top %d prediction precision: %.3f" %(topk,np.sum(result[:,2] == 1.0) / topk))
        np.savetxt(output_path + "/" + "anomaly_scores.txt", result, delimiter = ",")
    else:
        result = np.concatenate((top_index.reshape(topk,1), anomalyscores[top_index].reshape(topk,1)),axis = 1)
        #print("top %d prediction precision: %.3f" %(topk,np.sum(result[:,2] == 1.0) / topk))
        np.savetxt(output_path + "/" + "anomaly_scores.txt", result, delimiter = ",")
    
    # explain the outliers
    if is_explain:
        fimportance = score_in_chains(cf, X_explain)
        if projection:
            explain_object = Explanation(X,
                             feature_names =feature_names,
                             explanation_type = explain_method, 
                             projdim = projdim)
            ex = explain_object.explain(fimportance,result)
            X_transform = np.array(ex)
        else:
            X_transform = np.array(fimportance)
    
        # save the results with clustering indices
        X_result = np.concatenate((result[:,0:1], X_transform),axis =1)
        np.savetxt(output_path + "/" + "explanations.txt",X_result,delimiter=",")

        
if __name__ == '__main__':
    parameter_file = sys.argv[1]
    main(parameter_file)