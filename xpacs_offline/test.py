#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 19:59:32 2023

@author: dingxueying
"""

from xpacs import *
import pandas as pd
import matplotlib.pyplot as plt


def test():
    #loading a toy example
    #get the topk candidate rules
    cluster_number = 3
    cluster_id = 1
    mass_threshold = 0.5
    purity_threshold = 0.6
    dataset = "pkdd1998"
    
    with open('results/%s/cluster%d_idx%d_candidates.txt' %(dataset, cluster_number, cluster_id),'rb') as f:
        candidates = pickle.load(f)
    #print(candidates)
    scores = candidates[1]
    
    plt.scatter([i[0] for i in scores], [i[1] for i in scores], label = "current scores")
    plt.scatter([1.0],[1.0], label= "ideal")
    x = [i[0] for i in scores]
    y = [i[1] for i in scores]
    dist = np.square((np.array(x) - 1.0)) + np.square((np.array(y) - 1.0))
    plt.scatter(x[np.argmin(dist)], y[np.argmin(dist)], label = "least distant")
    plt.xlabel("mass/coverage")
    plt.ylabel("purity")
    plt.legend()
    plt.title("best_rule:%s" % str(candidates[0][np.argmin(dist)]))
    plt.savefig("results/%s/cluster%d_idx%d_candidates_score_plot.pdf" %(dataset, cluster_number, cluster_id))
    
    group0_anomaly = pd.read_csv('results/%s/cluster%d_idx%d_group_anomaly.txt'%(dataset, cluster_number, cluster_id),index_col = 0)
    group0_anomaly = group0_anomaly.reset_index()
    del group0_anomaly['index']
    normal_X = pd.read_csv('results/%s/normal_data.txt' % dataset)
    
    
    found_results = find_displaying_candidate_above_threshold(group0_anomaly,
                                             normal_X, 
                                             candidates=candidates[0],
                                             candidate_scores = candidates[1], 
                                             mass=mass_threshold, 
                                             purity=purity_threshold)
    print("results found:", found_results[0])
    print("scores found:", found_results[1])
         
    #calculcate the results for a particular rule
    #assume rule = found_results[0][0]
    ##rule = found_results[0][0] 
    rule = found_reults[0][0]
                   
    
    #other values needed
    cluster_number = 5
    cluster_id = 0
    mass_threshold = 0.5
    purity_threshold = 0.6
    
    with open('results/%s/cluster%d_idx%d_candidates.txt' %(dataset, cluster_number, cluster_id),'rb') as f:
        candidates = pickle.load(f)
    
    group0_anomaly = pd.read_csv('results/%s/cluster%d_idx%d_group_anomaly.txt'%(dataset, cluster_number, cluster_id),index_col = 0)
    group0_anomaly = group0_anomaly.reset_index()
    del group0_anomaly['index']
    x0 = group0_anomaly.iloc[0]
    cat_idx_lst = []
    for i in group0_anomaly.columns:
        if type(x0[i]) == str:
            cat_idx_lst.append(i)
    
    normal_X = pd.read_csv('results/%s/normal_data.txt' % dataset)

    mass_purity_score= get_mass_purity(group0_anomaly, normal_X, rule, feature_names = list(group0_anomaly.columns),
                                       cat_idx_lst = cat_idx_lst)
    print("Score:", mass_purity_score)
    
    

if __name__ == '__main__':
    test()