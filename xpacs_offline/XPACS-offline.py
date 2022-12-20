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


def get_quantile_info(X,index, log_dens, percentage, X_max, X_min):
    def find_all_thresholds(lst):
        quantile_lst =[]
        left_= 0
        right_ = 0
        for i in range(len(lst)):
            if i ==0:
                left_ = lst[i]
            else:
                if lst[i-1] +1 != lst[i]:
                    quantile_lst.append((left_,right_))
                    left_ = lst[i]
            right_ = lst[i]
        quantile_lst.append((left_,right_))
        return quantile_lst
    dens = np.exp(log_dens)
    candidate_lst = (dens > (percentage/100) * np.max(dens)).nonzero()[0].tolist()
    threshold_indices = find_all_thresholds(candidate_lst)
    true_threshold = []
    for i in threshold_indices:
        left_val = X[i[0]][0]
        if X_min[index] > X[i[0]][0]:
            left_val = X_min
        right_val = X[i[1]][0]
        if X_max[index] < X[i[1]][0]:
            right_val = X[i[1]][0]
        true_threshold.append([(index,(left_val, right_val))])
    #true_threshold = [[(index,(X[i[0]][0], X[i[1]][0]))] for i in threshold_indices]
    return true_threshold


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


def find_hyper_rectangles(index_lst,group_anomaly, X_max, X_min, is_show = False):
    rules = []
    #anomalies ={}
    for index in index_lst: 
        new_X = group_anomaly[:,index:index+1]
        val_range = (np.max(new_X) - np.min(new_X)) * 1e-8
        X_plot = np.linspace(np.min(new_X) - val_range ,np.max(new_X)+val_range, 1000)[:, np.newaxis]
        if is_show:
            fig, ax = plt.subplots()

        color  = "darkorange"
        kernel ="gaussian"
        lw = 2
        if is_show:
            ax.hist(new_X, bins=50, density=True)

        kde = KernelDensity(kernel=kernel, bandwidth=0.005).fit(new_X)
        log_dens = kde.score_samples(X_plot)
        if is_show:
            ax.plot(X_plot[:, 0],
                np.exp(log_dens),
                color=color,
                lw=lw,
                linestyle="-",
                label="kernel = '{0}'".format(kernel),)

        quantile = [80]
        color = ['red','limegreen','violet','deepskyblue']

        for i,c in zip(quantile,color):
            if is_show:
                ax.axhline(y=i/100 * np.max(np.exp(log_dens)), 
                       xmin=np.min(X_plot[:, 0]),
                       xmax=1, 
                       linestyle='--',
                       color = c, 
                       label = "%.2f" % (i/100)) 
            #print("Current threshold level %.2f" % (i/ 100))
            thresholds = get_quantile_info(X_plot,index,log_dens,i, X_max, X_min)
            #anomalies_above = anomaly_index_above_threshold(new_X,thresholds)
            rules.extend(thresholds)
            #anomalies[index][i] = anomalies_above
        if is_show:
            ax.legend(loc="upper left")
            ax.plot(new_X[:,0], -0.005 - 0.01  * np.random.random(new_X.shape[0]), "+k")
            ax.set_xlabel(index)
            #ax.plot(new_X[thresholds[1][0],0], -0.005 -0.01 * np.random.random(new_X[thresholds[1][1]].shape[0]),'+r')
            #plt.savefig("plots/xpacs_kde_index_%d.pdf" %index)
            plt.show()
           # plt.close()
    return rules


def generate_candidate(R_val, repeat):
    R_candidate = R_val
    all_possible_R_lst = []
    for possible_R in itertools.combinations(R_candidate, repeat+2):
        first_threshold = possible_R[0]
        for i in range(1,len(possible_R)):
            first_threshold = merge_thresholds(first_threshold, possible_R[i])
        all_possible_R_lst.append(first_threshold)
    return all_possible_R_lst


def remove_redundant_candidates(lst):
    b_set = set(tuple(x) for x in lst)
    b = [ list(x) for x in b_set ]
    return b


def union_lst(threshold_union_lst):
    union_results = []
    quantile_ = []
    value_ =[]
    cur_index = threshold_union_lst[0][0]
    for i in threshold_union_lst:
        quantile_.append('l')
        value_.append(i[1][0])
        quantile_.append('r')
        value_.append(i[1][1])
    sorted_value_, sorted_quantile_ = zip(*sorted(zip(value_, quantile_), key=lambda x: x[0]))
    for i in range(len(sorted_value_)-1):
        cur_value = sorted_value_[i]
        next_value = sorted_value_[i+1]
        if sorted_quantile_[i] == 'l' and sorted_quantile_[i+1] == 'r':
            union_results.append((cur_index,(cur_value, next_value)))
    return union_results


def merge_thresholds(threshold_1, threshold_2): 
    threshold_lst = threshold_1 + threshold_2
    result = []
    #find all available index in the threshold_lst:
    index_set = set()
    for item in threshold_lst:
        index_set.add(item[0])
    #for each index, we merge the boundaries if one is contained in other;
    for index in index_set:
        threshold_union_lst = [i for i in threshold_lst if i[0] == index]
        new_union_lst = union_lst(threshold_union_lst)
        result.extend(new_union_lst)
    return result


def find_all_candidates(R_candidates, group_anomaly, normal_X, mu, ms, rangeval = 5):
    R = []
    for repeat in range(rangeval):
        print("Current dimension: ", repeat+1)
        R_pure = []
        R_non_pure =[]
        
        for cur_rul in R_candidates:
            #print(cur_rul)
            print(cur_rul)
            anomaly_indx = anomaly_index_above_threshold(group_anomaly,cur_rul)
            normal_indx = anomaly_index_above_threshold(normal_X, cur_rul)
            mass = len(anomaly_indx)
            purity = len(normal_indx)
            if mass >= ms:
                if purity < mu:
                    #print("MASS: %d, PURITY: %d" %(mass, purity))
                    R_pure.append(cur_rul)
                else:
                    R_non_pure.append(cur_rul)
        if len(R_pure) != 0:
            R.extend(R_pure)
        if repeat == 0:
            R_val = R_pure+R_non_pure
        if repeat < rangeval -1:
            R_candidates = generate_candidate(R_val, repeat)
            R_candidates = remove_redundant_candidates(R_candidates)
        #print(len(R_candidates))
    final_R = remove_redundant_candidates(R)
    mass_purity_score =[]
    for i in final_R:
        mass, purity = get_mass_purity(group_anomaly, normal_X,i)
        mass_purity_score.append((mass, purity))
    return final_R, mass_purity_score



def find_top_k_candidate_above_threshold(group_anomaly,
                                         normal_X, 
                                         candidates,
                                         candidate_scores, 
                                         mass=0.3, 
                                         purity=0.91,
                                         k = 5):
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
    return (mass/group_anomaly.shape[0], 1- purity / normal_X.shape[0])



original_data = np.loadtxt("../drive_data_with_labels.txt",delimiter = ",")
X = original_data[:, 1:-1]
y = original_data[:,-1]
data_index = original_data[:,0].tolist()

scores = np.loadtxt("../drive_anomaly_scores.txt",delimiter = ",")
anomaly_scores = scores[:,1]
scores_index = scores[:,0].tolist()

anomaly_index = [int(item) for item in scores_index]
normal_index = [int(item) for item in data_index if item not in scores_index]

normal_X = X[normal_index]
np.save('results/normal.npy', normal_X)

#,normal_index = get_topk_prediction(scores, top_k=100)
top_k_features = X[anomaly_index]
top_k_ground_truth = y[anomaly_index]
top_k_scores = anomaly_scores
print("Top %d accuracy: %.4f" % (100, np.count_nonzero(top_k_ground_truth)/100))

X_min = np.min(X,axis = 0)
X_max = np.max(X,axis = 0)

outlier_data = np.loadtxt("../drive_feature_inference_results.projection.50.average.top100.result.txt",delimiter = ",")
# remove index
explanation_value = outlier_data[:,1:] 


dist_euclid = euclidean_distances(explanation_value)
mds = MDS(dissimilarity='precomputed', random_state=0)
# Get the embeddings
X_transform = mds.fit_transform(dist_euclid)

for cluster_number in range(2,10):
    #sil = []
    #label_lst = []
    kmeans = KMeans(n_clusters=cluster_number, random_state=0).fit(X_transform)
    labels = kmeans.labels_
    #label_lst.append(labels)
    #sil.append(silhouette_score(X_transform, labels, metric = 'euclidean'))
    for cluster_id in range(0,cluster_number):
        group0_anomaly = top_k_features[labels==cluster_id]
        explanation0_anomaly = explanation_value[labels == cluster_id]
        
        index_lst = np.argsort(np.mean(explanation0_anomaly,axis = 0))[::-1].tolist()[:10]
        
        rules = find_hyper_rectangles(index_lst,group0_anomaly, X_max, X_min)
        
        ms_percentage = 0.05
        pu_percentage = 0.05
        anomaly_group_shape = group0_anomaly.shape[0]
        normal_group_shape = normal_X.shape[0]
        
        R_candidates = deepcopy(rules)
        ms = ms_percentage * group0_anomaly.shape[0]
        mu = (1-pu_percentage) * normal_X.shape[0]
        #print(ms,mu)
        candidates = find_all_candidates(R_candidates, group0_anomaly, normal_X, mu= mu, ms=ms)
        #print(candidates[1])
        #write function
        with open('results/cluster%d_idx%d_candidates.txt' %(cluster_number, cluster_id),'wb') as f:
            pickle.dump(candidates, f)
        np.save('results/cluster%d_idx%d_group_anomaly.npy'%(cluster_number, cluster_id), group0_anomaly)
        
