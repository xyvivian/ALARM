"""
Look out algortihm, reference:
@misc{lookout,
  doi = {10.48550/ARXIV.1710.05333}, 
  url = {https://arxiv.org/abs/1710.05333},
  author = {Gupta, Nikhil and Eswaran, Dhivya and Shah, Neil and Akoglu, Leman and Faloutsos, Christos},
  keywords = {Social and Information Networks (cs.SI), FOS: Computer and information sciences, FOS: Computer and information sciences}, 
  title = {LookOut on Time-Evolving Graphs: Succinctly Explaining Anomalies from Any Detector}, 
  publisher = {arXiv}, 
  year = {2017}, 
  copyright = {arXiv.org perpetual, non-exclusive license}
}   
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from sklearn.preprocessing import LabelEncoder   
from collections import Counter
from itertools import combinations
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm
from copy import deepcopy
sys.path.append("../xstream")
from xstream import *
import os
import json
import sys
sys.path.append("..")


class Parameters():
    def __init__(self, 
                 json_file_name,
                 ):
        with open(json_file_name, 'r') as openfile:
            json_ = json.load(openfile)
        self.dataset_name = json_["dataset_name"]
        self.has_label = json_["has_label"]
        self.top_features = json_["top_features"]
        self.load_saved = json_["load_saved"]
        self.input_file = json_["input_file"]
        self.use_label = json_["use_label"]


#encode the categorical features to numbers with LabelEncoder
def encode_catgorical_column(data,column):
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    return le


""" Definition of Plot """
class Plot:
	def __init__( self, id ):
		self.id = id
		self.value = 0.0

	def get_id( self ): # Unique identifier of the plot
		return self.id

	def get_value( self ): # Total influence of the plot
		return self.value

	def update_value( self, value ):
		self.value = value

def get_topk_prediction(scores, top_k):
    ind = np.argpartition(scores, -top_k)[-top_k:]
    return ind


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

# Get scatter plot outlier scores and figure
def get_scores(original_data, feature_X, feature_Y, make_plot=True, get_log = False):
    data = original_data[:,(feature_X, feature_Y)]
    #print(data.shape)
    #you need to feed a subset of features, retraining is required
    new_model = XStream()
    new_model.fit(data)
    #print("We acquire the anomaly scores")
    
    scores = new_model.predict_proba(data)
    scores = minmax_scale(scores)
    ids = list(range(data.shape[0]))
    tuples = [(ids[i], scores[i]) for i in range(0, len(ids))]
    scores = sorted( tuples, key = lambda x: x[1], reverse = True )
    return scores

def normal_anomaly_idx_split(score_data, data_index):
    """
    split index of normal and anomaly points
    :param score_data: anomaly scores data
    :param data_index: original values
    :return: return split of indexes
    """
    scores_index = list(score_data.index)
    anomaly_index = [int(item) for item in scores_index]
    normal_index = [int(item) for item in data_index if item not in scores_index]
    #create anomaly columns
    anomaly_index_col = []
    for item in data_index:
        if item not in scores_index:
            anomaly_index_col.append(0)
        else:
            anomaly_index_col.append(1)
    return normal_index, anomaly_index,anomaly_index_col

def findPairs(n): 
    return list(Counter(combinations(n, 2)))


def print_format(ls):
    str_= str(ls)
    str_ = str_.replace(" ","")
    str_ = str_.replace(",","_")
    str_ = str_[1:-1]
    return str_

def str_to_tuple(val):
    sep = val.split(",")
    if len(sep) >1:
        i = sep[0]
        i = i.replace("('","")
        i = i.replace("'","")
        j = sep[1]
        j = j.replace(" ","")
        j = j.replace("'","")
        j = j.replace(")","")
        tup = (i, int(float(j)))
        return tup
    else:
        return None
    
def assign_scores(cluster_rank, outliers):
    plot_best = {}
    for plot_n,plot in enumerate(cluster_rank):
        for outlier in plot:
            idx = outlier[0]
            score = outlier[1]
            if idx in outliers: 
                if idx not in plot_best.keys():
                    plot_best[idx]= (plot_n,score)
                elif plot_best[idx][1] <= score:
                    plot_best[idx] = (plot_n,score)
    return plot_best


def LookOut(budget,cluster_rank_plot,sorted_plots,best_graphs,outliers):
    budget_best_graphs = best_graphs[0:budget]
    budget_plots = []
    for plot,score in sorted_plots:
        if budget > 0:
            budget_plots.append(cluster_rank_plot[plot])
            budget-=1
        else: break
    plot_max = {}
    #print(budget_plots)
    for key,value in assign_scores(budget_plots,outliers).items():
        anomaly_id = key
        plot_id = sorted_plots[value[0]][0]
        score = value[1]
        if plot_id not in plot_max.keys():
            plot_max[plot_id] = [score,[anomaly_id]]
        else:
            plot_max[plot_id][0] += score
            plot_max[plot_id][1] += [anomaly_id]
    return plot_max,budget_best_graphs


def plot(budget,
         cluster_rank_plot,
         sorted_plots,
         best_graphs,
         cluster,
         outliers,
         X_index,
         cluster_feature_pairs,
         feature_names,
         cat_dim_lst,
         encoder_mapping,
         X_encoded,
         anomaly_X):
    lo = LookOut(budget,cluster_rank_plot,sorted_plots,best_graphs,outliers)
    #lo[0] is the max_explained values
    #lo[1] is the graphs until the maximum budgets
    figure_list = []
    for plot in lo[1]:
        score = lo[0][plot][1]
        x,y = cluster_feature_pairs[cluster][plot]
        dfx = pd.DataFrame(X_index)
        color_idx = []
        for i in dfx[0]:
            if i not in list(anomaly_X.index):
                color_idx.append(0)
            else:
                if i in score:
                    color_idx.append(1)
                else:
                    if i in outliers:
                        color_idx.append(2)
                    else:
                        color_idx.append(3)
        
        dfx["color"] = color_idx # [0 if i not in list(anomaly_X.index) else 1 if i in score else 2 for i in dfx[0]]
        newPal = {0 :'#696969' , 1 :'#DC143C', 2 :'#3CB371'}
        fig = plt.figure(figsize = (4,3),dpi = 400)
        ax = fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #features = np.arange(X_encoded.shape[1])
        #features = np.load("data/data/dataset_features.npy",allow_pickle=True)
        plt.xlabel(feature_names[x])
        if feature_names[x] in cat_dim_lst:
            plt.xticks([encoder_mapping[feature_names[x]][i] for i in encoder_mapping[feature_names[x]].keys()],\
                       list(encoder_mapping[feature_names[x]].keys()),rotation=20) 
        
        plt.ylabel(feature_names[y])
        if feature_names[y] in cat_dim_lst:
            plt.yticks([encoder_mapping[feature_names[y]][i] for i in encoder_mapping[feature_names[y]].keys()],\
                       list(encoder_mapping[feature_names[y]].keys()),rotation=20) 
        #subset X,y
        #print(dfx.loc[dfx["color"] == 0].index)          
            
        plt.scatter(X_encoded[dfx.loc[dfx["color"] == 0].index,(x)],X_encoded[dfx.loc[dfx["color"] == 0].index,(y)],\
                    c='#696969',edgecolor='black',linewidth=0.3,alpha=0.7)
        plt.scatter(X_encoded[dfx.loc[dfx["color"] == 1].index,(x)],X_encoded[dfx.loc[dfx["color"] == 1].index,(y)],\
                    c='#DC143C',edgecolor='black',linewidth=0.3,alpha=0.85, s= 52, marker = "*")
        plt.scatter(X_encoded[dfx.loc[dfx["color"] == 2].index,(x)],X_encoded[dfx.loc[dfx["color"] == 2].index,(y)],\
                    c='#3CB371',edgecolor='black',linewidth=0.3,alpha=0.7, s = 30, marker = "s")
        figure_list.append(fig)
    return figure_list


def main(file_path, begin_cluster, end_cluster):
    parameters = Parameters(file_path)
    #load HPs
    dataset_name = parameters.dataset_name
    has_label = parameters.has_label
    top_features = parameters.top_features
    load_saved = parameters.load_saved
    use_label = parameters.use_label
    
    
    #load data, feature importances, y and etc.
    all_result = load_all_result("../data/%s/concatenate_result.txt" % dataset_name)
    original_data = load_all_result(parameters.input_file)
    data_index = original_data.index.tolist()
        
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
    scores = get_anomaly_score_data(all_result = all_result, use_label= use_label)
 
    normal_index, anomaly_index,anomaly_col = normal_anomaly_idx_split(scores, data_index)

    anomaly_col = pd.DataFrame(anomaly_col,columns = ["anomaly"])
    anomaly_col.index = X.index

    normal_X = X.iloc[normal_index]
    anomaly_X = X.iloc[anomaly_index]

    
    #transform data with labelencoder
    X_encoded = deepcopy(X)
    encoders = []
    for feat in cat_dim_lst:
        encoder = encode_catgorical_column(X_encoded,feat)
        encoders.append(encoder)
    X_encoded = np.array(X_encoded)

    #find clusters
    cluster_range = list(range(begin_cluster, end_cluster))
    clusters = [str(i) + " clusters" for i in cluster_range]
    cluster_int_index = [int(col[0]) for col in clusters if col.endswith("clusters")]
    cluster_indices = {}
    for cluster in clusters:
        for c in all_result[cluster].unique():
            cluster_indices[cluster, c] = list((all_result[all_result[cluster] == c].index)) 
    
    #get features
    cluster_features = {}
    
    for cluster in cluster_indices.keys():
        cluster_val = cluster_indices[cluster]
        features_list = np.argsort(np.mean(np.array(explanation_value.loc[cluster_val]),axis=0))[::-1][0:top_features]
        cluster_features[cluster] = features_list
    
    #get feature pairs
    cluster_feature_pairs = {}

    for features_list in cluster_features.keys():
        cluster_feature_pairs[features_list] = findPairs(cluster_features[features_list])
    
    if load_saved is False:
        clusters_rank_plot = {}
        #from tqdm import tqdm
        for feature_pairs in tqdm(cluster_feature_pairs.keys()):
            cluster_scores = []
            for pair in cluster_feature_pairs[feature_pairs]:
                cluster_scores.append(get_scores(X_encoded, pair[0],pair[1]))
            clusters_rank_plot[feature_pairs] = cluster_scores    
        
        clusters_rank_plot2 = {}
        for i in clusters_rank_plot.keys():
            clusters_rank_plot2[str(i)] = deepcopy(clusters_rank_plot[i])
        
        isExist = os.path.exists("../assets/%s" % dataset_name)
        if not isExist:
            os.makedirs("../assets/%s" % dataset_name)    
        
        #save the features!
        np.savez('../assets/%s/xstream%s.npz' % (dataset_name, print_format(cluster_int_index)), **clusters_rank_plot2, allow_pickle=True)
        df_xstreams_val = np.load('../assets/%s/xstream%s.npz' % (dataset_name, print_format(cluster_int_index)), allow_pickle=True)
    else:
        try:
            df_xstreams_val = np.load('../assets/%s/xstream%s.npz' % (dataset_name, print_format(cluster_int_index)), allow_pickle=True)
        except:
            print("pre-proceseed xstream file not found, consider switch load_save to False")
            exit()
            
    #encoder mapping
    encoder_mapping = {}
    for i,le in enumerate(encoders):
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        encoder_mapping[cat_dim_lst[i]] = le_name_mapping
    
    df_xstreams = {}
    for key in df_xstreams_val.keys():
        
        tup = str_to_tuple(key)
        if tup is not None:
            df_xstreams[tup] = df_xstreams_val[key]


    cluster_plot_scores = {}
    
    for cluster_ix in df_xstreams.keys():
        plot_scores = {}
        cluster_rank = df_xstreams[cluster_ix]
        outliers = cluster_indices[cluster_ix]
        for key,value in assign_scores(cluster_rank,outliers).items():
            anomaly_id = key
            plot_id = value[0]
            score = value[1]
            if plot_id not in plot_scores.keys():
                plot_scores[plot_id] = [score,[anomaly_id]]
            else:
                plot_scores[plot_id][0] += score
                plot_scores[plot_id][1] += [anomaly_id]
        sorted_plots = sorted(plot_scores.items(), key=lambda item: item[1][0],reverse=True)
        best_graphs = [plot[0] for plot in sorted_plots]
        cluster_plot_scores[cluster_ix]= (cluster_rank,sorted_plots,best_graphs)

    for key in cluster_plot_scores.keys():
        cluster_id = int(key[0][0])
        sub_cluster_id = key[1]
        a,b,c = cluster_plot_scores[key]

        for budget in range(1,6):
            figures = plot(budget,a,b,c,key, outliers = cluster_indices[key], X_index = data_index,
                           cluster_feature_pairs = cluster_feature_pairs,
                           feature_names =feature_names,
                           cat_dim_lst = cat_dim_lst,
                           encoder_mapping = encoder_mapping,
                           X_encoded = X_encoded,
                           anomaly_X = anomaly_X)
            for i,figure in enumerate(figures):
                #print(cluster_id, sub_cluster_id)
                fname = "../assets/"+dataset_name+"/{0}-{1}-{2}-{3}-{4}.png".format("lookout",cluster_id,sub_cluster_id,budget,i+1)
                figure.savefig(fname,bbox_inches = 'tight')
                plt.close(figure)
    

if __name__ == '__main__':
    parameter_file = sys.argv[1]
    begin_cluster = int(sys.argv[2])
    end_cluster = int(sys.argv[3])
    main(parameter_file, begin_cluster, end_cluster)
    