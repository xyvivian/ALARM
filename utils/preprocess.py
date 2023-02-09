import pandas as pd
import numpy as np

from copy import deepcopy
from sklearn.preprocessing import LabelEncoder

#encode the categorical features to numbers with LabelEncoder
def encode_catgorical_with_column(data, column):
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    return data,le






class DashboardData:
    def __init__(
        self,
        graph_data,
        dropdown_options,
        exploration_data,
        cluster_options,
        optimal_k,
        label_lst,
        feature_columns,
        feature_names,
        parallel_data,
        encoders,
        explanation_value
    ):
        self.graph_data = graph_data
        self.dropdown_options = dropdown_options
        self.exploration_data = exploration_data
        self.cluster_options = cluster_options
        self.optimal_k = optimal_k
        self.label_lst = label_lst
        self.feature_columns = feature_columns
        self.features = feature_names
        self.parallel_data =parallel_data
        self.encoders = encoders
        self.explanation = explanation_value


def prepare_data(
    all_result,
    original_data,
    anomaly_col,
    explanation_value,
    feature_names=None,
    cat_dim_lst = [], 
):
    #optimal k is a integer value
    optimal_k = all_result["optimal cluster"].iloc[0]
    cluster_index = [int(col[0]) for col in list(all_result.columns) if col.endswith("clusters")]
    cluster_col = [col for col in list(all_result.columns) if col.endswith("clusters")]
    graph_index = ['x','y'] + cluster_col + ['anomaly_scores']
    #feature column has [x,y, all clusters]
    feature_columns = ['x','y'] + cluster_col
    #graph data has [x,y,all clusters, scores]
    graph_data = all_result[graph_index]
    graph_data = graph_data.rename(columns={"anomaly_scores": "score"}) 
    graph_data.index = all_result.index
    #label list is the cluster columns -> convert to numpy array
    label_lst = np.array(all_result[cluster_col])
    dropdown_options = feature_names
    #original_data = np.array(original_data)
    exploration_data = pd.concat([original_data,anomaly_col], axis = 1)
    exploration_data.index = original_data.index
    #max_k = max(cluster_index)
    cluster_options =[]
    for ind in cluster_index:
        cluster_options.append(["cluster " + str(i) for i in range(ind)])
    # is the same as the exploration data, but each categories are encoded into labelencoder
    parallel_data =deepcopy(exploration_data)
    encoders = {}
    if feature_names != None:
        for vals in feature_names:
            if vals in cat_dim_lst:
                parallel_data, encode = (encode_catgorical_with_column(parallel_data,vals))
                encoders[vals] = encode     
    dashboard_data = DashboardData(
        graph_data,
        dropdown_options,
        exploration_data,
        cluster_options,
        optimal_k,
        label_lst,
        feature_columns,
        feature_names,
        parallel_data,
        encoders,
        explanation_value
    )
    #print(dashboard_data.cluster_options)
    return dashboard_data