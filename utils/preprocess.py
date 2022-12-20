import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
        feature_names
    ):
        self.graph_data = graph_data
        self.dropdown_options = dropdown_options
        self.exploration_data = exploration_data
        self.cluster_options = cluster_options
        self.optimal_k = optimal_k
        self.label_lst = label_lst
        self.feature_columns = feature_columns
        self.features = feature_names


def prepare_data(
    data_transformed,
    data_MDS,
    outlier_data,
    score_data,
    original_data,
    feature_names=None,
    max_k=10,
):
    if feature_names is None:
        feature_names = ["feature" + str(i) for i in range(1, outlier_data.shape[1]+1)]

    assert outlier_data.shape[1] == len(feature_names)

    sil = []
    label_lst = []
    for i in range(2, max_k):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(data_transformed)
        labels = kmeans.labels_
        label_lst.append(labels)
        sil.append(silhouette_score(data_transformed, labels, metric="euclidean"))
        col_name = str(i) + " clusters"
        data_MDS[col_name] = labels

    optimal_k = np.argmax(sil) + 2

    cluster_data = data_MDS
    feature_columns = cluster_data.columns
    cluster_data.index = outlier_data.index

    graph_data = pd.merge(cluster_data, outlier_data, how="left", on="index")
    scores = pd.DataFrame(score_data[:, :2], columns=["index", "score"]).set_index(
        "index"
    )
    graph_data = pd.merge(graph_data, scores, how="left", on="index")
    dropdown_options = feature_names
    exploration_data = pd.DataFrame(
        original_data, columns=["index"] + feature_names + ["anamoly"]
    ).set_index("index")
    cluster_options = ["cluster" + str(i) for i in range(max_k)]
    dashboard_data = DashboardData(
        graph_data,
        dropdown_options,
        exploration_data,
        cluster_options,
        optimal_k,
        label_lst,
        feature_columns,
        feature_names
    )
    return dashboard_data
