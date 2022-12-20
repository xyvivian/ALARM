import seaborn as sns
import pandas as pd
import numpy as np

shap = pd.read_csv("input/SHAPxstream.csv").iloc[:,:-1]

data = pd.read_csv("input/CleanEnrichedData.csv")

scores = pd.read_csv("input/enriched_altIP_with_scores.csv")[["xstream_scores"]]

data_scores = data
data_scores["scores"] = scores

anomaly_index = data_scores.sort_values(by=["scores"],ascending=False).head(100).index

data_scores["label"] = [1 if idx in anomaly_index else 0  for idx in data_scores.index] 

shap["label"] = [1 if idx in anomaly_index else 0  for idx in shap.index] 

df_shap = shap[shap["label"]==1].iloc[:,:-1].reset_index()
drive_anomaly_scores = data_scores[data_scores["label"]==1][["scores","label"]].reset_index()
drive_data_with_labels = data_scores.loc[:, data_scores.columns != "scores"].reset_index()

drive_anomaly_scores.to_csv('data/drive_anomaly_scores.txt',index=False,header=False)
drive_data_with_labels.to_csv('data/drive_data_with_labels.txt',index=False,header=False)
df_shap.to_csv('data/drive_feature_inference_results.projection.50.average.top100.result.txt',index=False,header=False)

np.save('data/dataset_features', np.array(drive_data_with_labels.columns[1:-1]))