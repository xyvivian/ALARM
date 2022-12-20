from utils import preprocess, utils

data, X, y, data_index = utils.get_driver("data_cancer/drive_data_with_labels.txt")
outlier_data, shap_inference = utils.get_inference(
    "data_cancer/drive_feature_inference_results.projection.50.average.top100.result.txt", feature_names=None)
score_data = utils.get_anomaly_score_data("data_cancer/drive_anomaly_scores.txt")

normal_index, anomaly_index = utils.normal_anomaly_idx_split(shap_inference, data_index)

normal_X = X[normal_index]
top_k_features = X[anomaly_index]

data_transformed, data_MDS = utils.get_mds(shap_inference)

DASHBOARD_DATA = preprocess.prepare_data(data_transformed, data_MDS, outlier_data, score_data, data, feature_names=None, max_k=10)
