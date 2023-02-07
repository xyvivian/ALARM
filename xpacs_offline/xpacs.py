import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt
import sklearn.datasets as dt
import seaborn as sns         
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import itertools
import pickle
from copy import deepcopy
import random
import sys
import os
import json
from sklearn.preprocessing import LabelEncoder


#encode the categorical features to numbers with LabelEncoder
def encode_catgorical_column(data):
    le = LabelEncoder()
    data = le.fit_transform(data)
    return data,le


def get_real_quantile_info(X,index, log_dens, percentage, X_max, X_min):
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
            left_val = X_min[index]
        right_val = X[i[1]][0]
        if X_max[index] < X[i[1]][0]:
            right_val = X_max[index]
        true_threshold.append([(index,(left_val, right_val))])
    #true_threshold = [[(index,(X[i[0]][0], X[i[1]][0]))] for i in threshold_indices]
    return true_threshold


def get_cat_quantile_info(index,log_dens,mapping):
    return [[(index,(mapping[np.argmax(log_dens)],mapping[np.argmax(log_dens)]))]]



def anomaly_index_above_threshold(X,threshold,feature_names,cat_idx_lst):
    anomaly_indices = {}
    for index,thres in threshold:
        if index not in anomaly_indices.keys():    
            if feature_names[index] not in cat_idx_lst:
                anomaly_indices[index] = (np.where((X[feature_names[index]] >= thres[0]) &\
                                                   (X[feature_names[index]] <= thres[1]))[0].tolist())    
            else:
                anomaly_indices[index] = X.index[X[feature_names[index]] == thres[0]].tolist()
    ads = list(anomaly_indices.values())
    if ads == []:
        return []
    elif len(ads)> 1:
        return list(set(ads[0]).intersection(*ads))
    else:
        return ads[0]


def find_hyper_rectangles(index_lst,group_anomaly, X_max, X_min, cat_idx_lst,is_show = False):
    kernel ="gaussian"
    color  = "darkorange"
    color_bar = ['red','limegreen','violet','deepskyblue']
    lw = 2
    rules = []
    labelencoders = []
    #anomalies ={}
    index_names = list(group_anomaly.columns)
    quantile = [80]
    for index in index_lst: 
        if index_names[index] in cat_idx_lst:
            is_cat = True
            new_X = deepcopy(group_anomaly[index_names[index]])
            new_X, encoder = encode_catgorical_column(new_X)
            labelencoders.append(encoder)
            hist = np.histogram(new_X, bins=len(group_anomaly[index_names[index]].unique()), density=True, weights=None)
            if is_show:
                plt.clf()
                _ = plt.hist(new_X, bins=len(group_anomaly[index_names[index]].unique()))
                #le_name_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                plt.xticks(encoder.transform(encoder.classes_),encoder.classes_)
                plt.show()
            log_dens = hist[0]
            thresholds = get_cat_quantile_info(index,log_dens,encoder.classes_)
            rules.extend(thresholds)
        else:
            is_cat = False
            new_X = group_anomaly[index_names[index]]
            new_X = np.array(new_X)
            new_X = new_X.reshape((new_X.shape[0],1))
            val_range = (np.max(new_X) - np.min(new_X)) * 1e-8
            X_plot = np.linspace(np.min(new_X) - val_range ,np.max(new_X)+val_range, 1000)[:, np.newaxis]
            if is_show:
                plt.clf()
                fig, ax = plt.subplots()
                ax.hist(new_X, bins=50, density=True)
            if np.max(new_X) - np.min(new_X) > 0.0:
                bandwidth = (np.max(new_X) - np.min(new_X))*0.02
            else:
                bandwidth = 1
            kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(new_X)
            log_dens = kde.score_samples(X_plot)
            if is_show:
                ax.plot(X_plot[:, 0],
                    np.exp(log_dens),
                    color=color,
                    lw=lw,
                    linestyle="-",
                    label="kernel = '{0}'".format(kernel),)
            for i,c in zip(quantile,color_bar):
                if is_show:
                    ax.axhline(y=i/100 * np.max(np.exp(log_dens)), 
                           xmin=np.min(X_plot[:, 0]),
                           xmax=1, 
                           linestyle='--',
                           color = c, 
                           label = "%.2f" % (i/100)) 
                thresholds = get_real_quantile_info(X_plot,index,log_dens,i, X_max, X_min)
                rules.extend(thresholds)
                plt.show()
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
    cur_value = sorted_value_[0]
    next_value = sorted_value_[-1]
    if sorted_quantile_[0] == 'l' and sorted_quantile_[-1] == 'r':
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


def find_all_candidates(R_candidates, group_anomaly, normal_X, mu, ms,cat_idx_lst,anomaly_shape, rangeval = 5):
    R = []
    feature_names = list(group_anomaly.columns)
    for repeat in range(rangeval):
        print("Current dimension: ", repeat+1)
        R_pure = []
        R_non_pure =[]
        for cur_rul in R_candidates:
            #print(cur_rul)
            anomaly_indx = anomaly_index_above_threshold(group_anomaly,cur_rul,feature_names, cat_idx_lst)
            normal_indx = anomaly_index_above_threshold(normal_X, cur_rul,feature_names, cat_idx_lst)
            mass = len(anomaly_indx) / anomaly_shape
            if (len(anomaly_indx) + len(normal_indx)) == 0:
                purity = 0.0
            else:
                purity = len(normal_indx) / (len(anomaly_indx) + len(normal_indx))
            
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
        mass, purity = get_mass_purity(group_anomaly, normal_X,i,feature_names,cat_idx_lst)
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
    k = min([len(val),k])
    for i in range(k):
        found_index.append(val[i][0]) 
    return [candidate_lst[i] for i in found_index], [mass_purity_lst[i] for i in found_index]


def find_displaying_candidate_above_threshold(group_anomaly,
                                         normal_X, 
                                         candidates,
                                         candidate_scores, 
                                         mass=0.3, 
                                         purity=0.91):
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
    val1 = sorted(indexed_mass_purity_lst,key=lambda sl: (-sl[1][0]))
    val2 = sorted(indexed_mass_purity_lst,key=lambda sl: (-sl[1][1]))
    if len(val1) ==1:
        found_index = []
        found_index.append(val1[0][0])
    elif len(val1) == 2:
        found_index = []
        found_index.append(val1[0][0])
        found_index.append(val1[1][0])
    elif len(val1) == 3:
        found_index = []
        for i in range(3):
            found_index.append(val1[i][0])
    else:
        max_val1 = val1[0][0]
        max_val2 = val2[0][0]
        if max_val1 == max_val2:
            max_val2 = val2[1][0]
        
        max_val3 = random.choice(val1)[0]
        while (max_val3 == max_val1) or (max_val3 == max_val2):
            max_val3 = random.choice(val1)[0]
        found_index = [max_val1, max_val2, max_val3]
    return [candidate_lst[i] for i in found_index], [mass_purity_lst[i] for i in found_index]



def get_mass_purity(group_anomaly, normal_X, candidate_rule,feature_names,cat_idx_lst):
    anomaly_indx = anomaly_index_above_threshold(group_anomaly,candidate_rule,feature_names,cat_idx_lst)
    normal_indx = anomaly_index_above_threshold(normal_X, candidate_rule,feature_names,cat_idx_lst)
    mass = len(anomaly_indx)
    purity = len(normal_indx)
    #print(mass,purity)
    if len(normal_indx) == 0:
        purity = 0.0
    else:
        purity = purity / (len(anomaly_indx) + len(normal_indx))
    return (mass/group_anomaly.shape[0], 1- purity )


def load_all_result(path):
    """
    Read all results from path
    """
    return pd.read_csv(path, delimiter=",",index_col = "index")


def get_inference(all_result, X, feature_names=None):
    """
    Gets  explain value information
    :param result: all_results procesed by Xstream
    :param feature_name: optional feature names
    :return:
    """
    col_names = list(all_result.columns)
    if feature_names is None:
        feature_names = list(X.columns)
  
    explain_ = [feat + "_ex" for feat in feature_names]     

    #print(feature_names)
    assert len(explain_) == len(feature_names)
    assert len(explain_) == len(list(X.columns))
    for i in explain_:
        assert(i in col_names)
   
    outlier_explain = all_result[explain_]
    outlier_explain.index.names = ["index"]
    return outlier_explain

def get_anomaly_score_data(all_result,use_label):
    """
    get xstream anomaly scores
    :param result: all_results procesed by Xstream
    :return: array with anomaly scores
    """
    if use_label == False:
        assert "anomaly_scores" in list(all_result.columns)
        return  all_result["anomaly_scores"]
    else:
        assert "label" in list(all_result.columns)
        return all_result["label"]



class Parameters():
    def __init__(self, 
                 json_file_name,
                 ):
        with open(json_file_name, 'r') as openfile:
            json_ = json.load(openfile)
        self.dataset_name = json_["dataset_name"]
        self.topk = json_["topk"]
        self.has_label = json_["has_label"]
        self.top_features = json_["top_features"]
        self.input_file = json_["input_file"]
        self.use_label = json_["use_label"]
        
        
def main(file_path):
    parameters = Parameters(file_path)
    dataset_name = parameters.dataset_name
    topk = parameters.topk
    has_label = parameters.has_label
    top_features = parameters.top_features
    use_label = parameters.use_label
    
    isExist = os.path.exists("results/%s" % dataset_name)
    if not isExist:
       os.makedirs("results/%s" % dataset_name) 

    ms_percentage = 0.05
    pu_percentage = 0.05

    all_result = load_all_result("../data/%s/concatenate_result.txt" % dataset_name)
    original_data = load_all_result(parameters.input_file)
    if has_label:
        feature_names = list(original_data.columns)[0:-1]
    else:
        feature_names = list(original_data.columns)
        
    X = original_data[feature_names]
    data_index = list(X.index)
    cat_dim_lst = []
    for idx,ival in enumerate(X.iloc[0]):
        if type(ival) == str:
            cat_dim_lst.append(list(X.columns)[idx])
            
    explanation_value = get_inference(all_result= all_result, 
                                        X= X,
                                        feature_names=feature_names)
    scores = get_anomaly_score_data(all_result = all_result,use_label = use_label)
    anomaly_scores = np.array(scores)
    scores_index = list(scores.index)
    
    anomaly_index = [int(item) for item in scores_index]
    normal_index = [int(item) for item in data_index if item not in scores_index]
    
    #save normal data for future inference
    normal_X = X.iloc[normal_index]
    normal_X.to_csv("results/%s/normal_data.txt" % dataset_name,index=None, sep=',')
    
    #find max min of each features' values
    X_min = np.min(X,axis = 0)
    X_max = np.max(X,axis = 0)
    
    #find clusters
    cluster_index = [int(col[0]) for col in list(all_result.columns) if col.endswith("clusters")]
    
    for cluster_number in range(min(cluster_index), max(cluster_index)+1):
        for cluster_id in range(0, cluster_number):
            index = all_result.loc[all_result[str(cluster_number) + " clusters"]== cluster_id].index
            group0_anomaly = X.iloc[index]
            group0_anomaly.to_csv('results/%s/cluster%d_idx%d_group_anomaly.txt'%(dataset_name,cluster_number, cluster_id),index=True, sep=',')
            group0_anomaly = group0_anomaly.reset_index()
            del group0_anomaly['index']
            explanation0_anomaly = explanation_value.loc[index]
            index_lst = np.argsort(np.mean(explanation0_anomaly,axis = 0))[::-1].tolist()[:top_features]
            rules = find_hyper_rectangles(index_lst,group0_anomaly, X_max, X_min, cat_dim_lst,is_show = False)
            R_candidates = deepcopy(rules)
            candidates = find_all_candidates(R_candidates,
                                     group0_anomaly,
                                     normal_X, 
                                     mu= (1- pu_percentage), 
                                     ms= ms_percentage,
                                     cat_idx_lst =cat_dim_lst,
                                     anomaly_shape = group0_anomaly.shape[0])
            with open('results/%s/cluster%d_idx%d_candidates.txt' %(dataset_name,cluster_number, cluster_id),'wb') as f:
                pickle.dump(candidates, f)
                
if __name__ == '__main__':
    parameter_file = sys.argv[1]
    main(parameter_file)
    