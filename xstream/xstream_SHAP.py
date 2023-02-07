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
import shap
from sklearn.manifold import MDS
from matplotlib import pyplot as plt
import sklearn.datasets as dt
import seaborn as sns         
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder
import os
import math
import random
import mmh3
import tqdm



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
        for i in range(self.nchains):
            chain = self.chains[i]
            scores += chain.score(projected_X, adjusted)
        scores /= float(self.nchains)
        return -scores


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
    #calculate the scores on splits of each chains
    for cindex in range(0, cf.nchains):
        c = cf.chains[cindex]
        score_c_AllDepths = c.score_all_depths(X_to_explain)

        ind_min = np.argmin(score_c_AllDepths,axis = 1)
        score_c = np.zeros((X_to_explain.shape[0],))
        for idx,ind in enumerate(ind_min):
            score_c[idx] = score_c_AllDepths[idx,ind]
        cSplitFeatures = c.fs
        
        for idx,ind in enumerate(ind_min):
            for split_ind, split_f in enumerate(cSplitFeatures[0:ind+1]):
                fused[idx,split_f,cindex] = 1
                score_in_c[idx,split_f] = score_in_c[idx,split_f] + score_c_AllDepths[idx,split_ind] #- score_c[idx]
                
        #if the feature is never used in any splits
        #instead of assigining it to zero, we assign a negative value equivalant to the mininum score
        #so this feature will not be the largest!
        for idx in range(fused.shape[0]):
            for split_f in range(fused.shape[1]):
                if np.sum(fused[idx,split_f,:]) == 0.0:
                    score_in_c[idx,split_f] = np.min(score_in_c[idx]) 
                    
        fimportance = np.zeros(X_to_explain.shape)
        for f in range(fimportance.shape[1]):
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
        self.R_ohe = self.get_Rohe(X)

        
    def explain(self,fimportance,result):
        explain = []
        for i in range(fimportance.shape[0]):
            if self.explanation_type == "average":
                explain.append(self.avg_projection(fimportance[i,:]))
            elif self.explanation_type == "random_walk":
                explain.append(self.random_walk_projection(fimportance[i,:]))
            elif self.explanation_type == "hits":
                explain.append(self.hits_projection(fimportance[i,:]))
            elif self.explanation_type == "signed_hits":
                explain.append(self.signed_hits_projection(fimportance[i,:]))
        #now explain should equal to topk times total_feature_dimensions
        #if we have one-hot-encoding, we should only select the feature index with "1".
        explain_distinct = []
        for i in range(len(explain)):
            #X_index = int(result[i,0])
            if type(self.X) == np.ndarray:
                X_original_val = self.X[i]
            else:
                X_original_val = self.X.iloc[i]
            sum_feature = np.zeros((self.ndim,))
            for j in range(self.ndim):
                if self.types[j]:
                    if type(self.X) == np.ndarray:
                        distinctDF = np.unique(self.X[:,j]).tolist()
                    else:
                        distinctDF = list(self.X[self.X.columns[j]].unique())
                    X_feature_idx = distinctDF.index(X_original_val[j])
                    if j == 0:
                        begin_indx = 0
                    else:
                        begin_indx = int(np.sum(self.num_unique[0:j]))
                    sum_feature[j] =  explain[i][begin_indx + X_feature_idx]
                else:
                    if j == 0:
                        begin_indx =0
                    else:
                        begin_indx = int(np.sum(self.num_unique[0:j]))
                    sum_feature[j] = explain[i][begin_indx:begin_indx + 1]
            explain_distinct.append(sum_feature)
        return np.array(explain_distinct)
        
    def get_Rohe(self,X):
        self.num_unique = np.zeros((self.ndim,1))
        for i in range(self.ndim):
            if self.types[i]:
                if type(X) == np.ndarray:
                    self.num_unique[i] = np.unique(X[:,i]).shape[0]
                else:
                    self.num_unique[i] = len(X[X.columns[i]].unique())
            else:
                self.num_unique[i] = 1
        totalAfterOHE = self.num_unique.sum() 
        R_ohe = np.zeros((self.projdim, int(totalAfterOHE))) 
        keys = np.arange(0,self.projdim,1)
        ind = 0
        for i in range(self.ndim):
            if self.types[i]:
                if type(X) == np.ndarray:
                    distinctDF = np.unique(X[:,i])
                else:
                    distinctDF = X[X.columns[i]].unique()
                outrdd = []
                for j in distinctDF:
                    outrdd.append((j,np.array([_hash_string(k, "%s%s%s" %(self.feature_names[i],'.',j)) for k in keys])))
                for v in outrdd:
                    R_ohe[:,ind] = v[1].T
                    ind = ind + 1
            else:
                f = self.feature_names[i]
                R_ohe[:,ind] = np.array([_hash_string(k, f) for k in keys])
                ind = ind + 1  
        return R_ohe


    def avg_projection(self,fimportance):    
        fimpr = deepcopy(fimportance) # which one to use
        totalprojperfeat = np.sum(self.R_ohe, axis=0)
        #print(totalprojperfeat)
        #totalfeatureperproj = np.sum(self.R_ohe, axis=1)
        scaledR = self.R_ohe.copy()
        for p in range(self.R_ohe.shape[0]):
            scaledR[p,:] = scaledR[p,:] * fimpr[p]
        origfimportance = np.sum(scaledR, axis=0) / totalprojperfeat
        return origfimportance


    def random_walk_projection(self,fimportance,alpha = 0.15,repeat_times = 20):
        # attribution to original features
        # 1. BY RANDOM WALKS
        dr = np.random.rand(self.R_ohe.shape[1],1)
        pr = np.random.rand(self.R_ohe.shape[0],1)
        sumr = dr.sum()+pr.sum()
        dr = dr / sumr
        pr = pr / sumr
        fimpr = fimportance.copy()
        newR = self.R_ohe.copy()
        #Option 1: normalize the features
        if np.min(fimpr) < 0:
            fimpr = fimpr + np.abs(np.min(fimpr))  
        fimpr = fimpr / sum(fimpr) # normalize so sum to 1, a prob dist.n
        fimpr = fimpr.reshape(newR.shape[0],1)
        sums = newR.sum(axis=0,keepdims=1)
        sums[sums==0] = 1
        A = newR.copy() / sums
        B = newR.copy().T
        sums = B.sum(axis=0,keepdims=1)
        sums[sums==0] = 1
        B = B / sums
        for i in range(repeat_times):
            drnew = alpha * np.dot(B, pr)
            prnew = alpha * np.dot(A, dr) + (1-alpha) * fimpr
            denom = (drnew.sum()+prnew.sum())
            #print(denom) # should be 1
            dr = drnew.reshape(self.R_ohe.shape[1],1) / denom
            pr = prnew.reshape(self.R_ohe.shape[0],1) / denom
        return (dr/dr.sum())

    def hits_projection(self,fimportance,alpha=0.5,repeat_time = 20):
        newR = self.R_ohe.copy()
        fimpr = fimportance.copy() 
        # Shift so that minimum is zero
        if np.min(fimpr) < 0:
            fimpr = fimpr + np.abs(np.min(fimpr))

        fimpr = fimpr / LA.norm(fimpr,2)
        fimpr = fimpr.reshape(self.R_ohe.shape[0],1)
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

    def signed_hits_projection(self,fimportance, epsilon = 0.1, repeat_time = 20):
        dr = np.ones(self.R_ohe.shape[1]) * epsilon # importance
        pr = np.ones(self.R_ohe.shape[0]) * epsilon # authority/accurateness
        B = self.R_ohe.copy().T
        sums = B.sum(axis=0,keepdims=1)
        sums[sums==0] = 1
        scaledR = self.R_ohe.copy()
        scaledRtrans = scaledR.T / sums
        for i in range(repeat_time):
            dr = np.dot(scaledRtrans, pr)
            dr = dr.reshape(scaledR.shape[1],1) / LA.norm(dr,2) # abs(dr.sum())
            pr = np.dot(scaledRtrans.T, dr)
            pr = pr.reshape(scaledR.shape[0],1) / LA.norm(pr,2) # abs(pr.sum())
        return dr

def main(dataset):
    projdim = 10
    nchains = 200
    depth = 20
    projection = False
    input_file = "../data/%s/generated_synthetic.txt" %dataset
    explain_type = "random_walk"
    output_path = "../data/%s/xstream_pred" % dataset
    is_explain = True
    explain_method = "random_walk"
    has_label = True
    use_label = False
    #create output directory
    isExist = os.path.exists(output_path)
    if not isExist:
        os.makedirs(output_path)  

    #load the data 
    data = pd.read_csv(input_file,delimiter=",",index_col=0)
    
    feature_names = list(data.columns)[0:-1]
    label_name = list(data.columns)[-1]
    Y = data[label_name]
    X = data[feature_names]

    cat_features = []
    flt_features = []
    for i in feature_names:
        if type(X.loc[0][i]) == str:
            cat_features.append(i)
        else:
            flt_features.append(i)
        
    cat_X = X[cat_features]
    flt_X = X[flt_features]
    transformed_X = pd.concat((cat_X, flt_X), axis = 1)
    

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(cat_X)
    trans_X = enc.transform(cat_X)
    trans_X = np.concatenate((trans_X.toarray(),np.array(flt_X)),axis=1)
        

    cf = Chains(k=projdim, nchains=nchains, depth=depth, projection = projection)
    projected_X = cf.get_projection(trans_X,feature_names)
    cf.fit(projected_X)
    anomalyscores = cf.score(projected_X)
    if has_label:
        ap = average_precision_score(Y, anomalyscores) 
        auc = roc_auc_score(Y, anomalyscores)
        print("xstream: AP =", ap, "AUC =", auc)

    anomalyscores = (anomalyscores - anomalyscores.min(axis=0)) / (anomalyscores.max(axis=0) - anomalyscores.min(axis=0))
    #anomaly indices
    anomaly_indices = np.where(Y == 1)
    X_explain = trans_X[anomaly_indices]
    

    explainer = shap.Explainer(cf.score, X_explain)
    shap_values = explainer(X_explain)
    np.savetxt(output_path + "_" + "xstream_no_projection_shap.txt", shap_values.values, delimiter = ",")

    explanations = []
    for ad_idx,ad in enumerate(anomaly_indices):
        explain= np.zeros((X.shape[1],))
        anomaly= trans_X[ad]
        total_cat_dim = sum([len(cat_X[i].unique()) for i in list(cat_X.columns)])
        for true_index,i in enumerate(feature_names):
            if i in flt_features:
                idx = flt_features.index(i) + total_cat_dim
                explain[true_index] = shap_values.values[ad_idx][idx]
            elif i in cat_features:
                idx = cat_features.index(i)
                #one-hot-encoding of previous mapped features
                prev_sum = sum([len(cat_X[j].unique()) for j in list(cat_X.columns)[0:idx]])
                #current one-hot-encoding
                ohe = anomaly[prev_sum : prev_sum + len(list(cat_X[i].unique()))]
                sp = shap_values.values[ad_idx][prev_sum : prev_sum + len(list(cat_X[i].unique()))]
                explain[true_index] = np.sum(np.multiply(ohe,sp))
        explanations.append(explain)
        
    np.save("../results/%s/xstream_shap_explain.npy" % dataset,np.array(explanations))
    
if __name__ == '__main__':
    parameter_file = sys.argv[1]
    main(parameter_file)   
