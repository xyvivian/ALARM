import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import copy
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
import sklearn.datasets as dt
import seaborn as sns         
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import itertools
import pickle
from copy import deepcopy

def anomaly_index_above_threshold(X,threshold):
    anomaly_indices = {}
    for index,thres in threshold:
        if index not in anomaly_indices.keys():      
            anomaly_indices[index] = (np.where((X[:,index] >= thres[0]) & (X[:,index] <= thres[1]))[0].tolist())    
        else:
            anomaly_indices[index].extend(np.where((X[:,index] >= thres[0]) & (X[:,index] <= thres[1]))[0].tolist())
        #print("At threshold (%.4f, %.4f), the satisfying X: " % (thres[0], thres[1]))
        #print(X[np.where((X >= thres[0]) & (X <= thres[1]))[0]])
    #print("Total count of anomalies above threshold: %d " % len(anomaly_indices))
    ads = list(anomaly_indices.values())
    return list(set(ads[0]).intersection(*ads))

def find_top_k_candidate_above_threshold(group_anomaly,
                                         normal_X, 
                                         candidates,
                                         candidate_scores, 
                                         mass=0.3, 
                                         purity=0.91,
                                         k = 3):
    candidate_lst = []
    mass_purity_lst = []
    for index, i in enumerate(candidate_scores):
        #anomaly_indx = anomaly_index_above_threshold(group_anomaly,i)
        #normal_indx = anomaly_index_above_threshold(normal_X, i)
        mass_i = i[0]
        purity_i = i[1]
        if mass_i >= mass and purity_i >= purity:
            candidate_lst.append(candidates[index])
            mass_purity_lst.append([mass_i, purity_i])
    if len(mass_purity_lst) ==0:
        return [], []
    indexed_mass_purity_lst = list(zip(list(range(len(mass_purity_lst))), mass_purity_lst))  
    val = sorted(indexed_mass_purity_lst,key=lambda sl: (-sl[1][0],-sl[1][1]))
    found_index = []
    for i in range(k):
        found_index.append(val[i][0]) 
    return [candidate_lst[i] for i in found_index], [mass_purity_lst[i] for i in found_index]


def get_mass_purity(group_anomaly, normal_X, candidate_rule):
    anomaly_indx = anomaly_index_above_threshold(group_anomaly,candidate_rule)
    normal_indx = anomaly_index_above_threshold(normal_X, candidate_rule)
    mass = len(anomaly_indx)
    purity = len(normal_indx)
    #print("Current MASS: %d, Current PURITY:%d " % (mass,purity))
    return (mass/group_anomaly.shape[0], 1- purity / normal_X.shape[0])


#loading a toy example
#get the topk candidate rules
cluster_number = 5
cluster_id = 0
mass_threshold = 0.5
purity_threshold = 0.6

with open('results/cluster%d_idx%d_candidates.txt' %(cluster_number, cluster_id),'rb') as f:
    candidates = pickle.load(f)

group0_anomaly = np.load('results/cluster%d_idx%d_group_anomaly.npy'%(cluster_number, cluster_id))
normal_X = np.load('results/normal.npy')


found_results = find_top_k_candidate_above_threshold(group0_anomaly, normal_X, candidates[0],candidates[1],k = 3,
                                                    mass= mass_threshold, purity = purity_threshold)
print("results found:", found_results[0])
print("scores found:", found_results[1])
     
#calculcate the results for a particular rule
#assume rule = found_results[0][0]
##rule = found_results[0][0] 
rule = [[7, (0.0165, 0.0241)]]
               
    
#other values needed
cluster_number = 5
cluster_id = 0
mass_threshold = 0.5
purity_threshold = 0.6

with open('results/cluster%d_idx%d_candidates.txt' %(cluster_number, cluster_id),'rb') as f:
    candidates = pickle.load(f)

group0_anomaly = np.load('results/cluster%d_idx%d_group_anomaly.npy'%(cluster_number, cluster_id))
normal_X = np.load('results/normal.npy')
mass_purity_score= get_mass_purity(group0_anomaly, normal_X, rule)
print("Score:", mass_purity_score)