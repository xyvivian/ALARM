import pandas as pd
import numpy as np

from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances


def get_driver(path):
    """
    Gets driver information
    :param path: path to driver file
    :return: X data_cancer from driver, y score from driver, data_index index of each value
    """
    original_data = np.loadtxt(path, delimiter=",")
    X = original_data[:, 1:-1]
    y = original_data[:, -1]
    data_index = original_data[:, 0].tolist()
    return original_data, X, y, data_index


def get_inference(path, feature_names=None):
    """
    Gets shap value information
    :param path: path to shap inference scores
    :param feature_names: optional feature names
    :return:
    """
    shap_inference = np.loadtxt(path, delimiter=",")
    if feature_names is None:
        feature_names = ["feature" + str(i) for i in range(shap_inference.shape[1])]

    assert shap_inference.shape[1] == len(feature_names)
    explanation_value = shap_inference[:, 1:]
    outlier_data = pd.DataFrame(shap_inference, columns=feature_names).set_index(
        "feature0"
    )
    outlier_data.index.names = ["index"]
    return outlier_data, explanation_value


def get_anomaly_score_data(path):
    """
    get xstream anomaly scores
    :param path: path
    :return: array with anomaly scores
    """
    return np.loadtxt(path, delimiter=",")


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


def normal_anomaly_idx_split(shap_inference, data_index):
    """
    split index of data_cancer and anomaly points
    :param shap_inference: shap inference file
    :param data_index: original values
    :return: return split of indexes
    """
    scores_index = shap_inference[:, 0].tolist()
    anomaly_index = [int(item) for item in scores_index]
    normal_index = [int(item) for item in data_index if item not in scores_index]
    return normal_index, anomaly_index
