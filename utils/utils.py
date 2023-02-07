import pandas as pd
import numpy as np

from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances


def get_driver(path, has_label=True):
    """
    Gets driver information
    :param path: path to driver file
    :param feature_names: the name of all features
    :return: X data_cancer from driver, y score from driver, data_index index of each value
    """
    original_data =  pd.read_csv(path,delimiter=",",index_col = "index")
    if has_label:
        feature_names = list((original_data.columns))[0:-1]
        label_name = list(original_data.columns)[-1]
        X = original_data[feature_names]
        y = original_data[label_name]
    else:
        feature_names = list(original_data.columns)
        X = original_data[feature_names]
    data_index = original_data.index.tolist()
    return X, data_index,feature_names

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


def get_anomaly_score_data(all_result):
    """
    get xstream anomaly scores
:   :param result: all_results procesed by Xstream
    :return: array with anomaly scores
    """
    assert "anomaly_scores" in list(all_result.columns)
    return  all_result["anomaly_scores"]


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