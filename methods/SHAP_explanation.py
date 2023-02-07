import os
import time
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import pandas as pd
import shap


class IsoForest():
    def __init__(self, n_estimators=256, max_samples='auto', **kwargs):
        # initialize
        self.isoForest = None
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.initialize_isoForest(**kwargs)


    def initialize_isoForest(self, seed=0, **kwargs):
        self.isoForest = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples
                                         , n_jobs=-1, random_state=seed, **kwargs)

    def fit(self, train_X):
        print("Starting training...")
        start_time = time.time()
        self.isoForest.fit(train_X.astype(np.float64))
        end_time = time.time() - start_time
        return end_time

    def predict_proba(self,test_X):
        #print("Starting prediction...")
        scores = (-1.0) * self.isoForest.decision_function(test_X.astype(np.float32))  # compute anomaly score
        #y_pred = (self.isoForest.predict(test_X.astype(np.float32)) == -1) * 1  # get prediction
        #auc = roc_auc_score(test_y, scores.flatten())
        #print("AUCROC: %.4f" % auc)
        return scores
    
    def predict(self,test_X,test_y):
        print("Starting prediction...")
        scores = (-1.0) * self.isoForest.decision_function(test_X.astype(np.float32))  # compute anomaly score
        #y_pred = (self.isoForest.predict(test_X.astype(np.float32)) == -1) * 1  # get prediction
        auc = roc_auc_score(test_y, scores.flatten())
        print("AUCROC: %.4f" % auc)
        return scores
    

dataset = "ids"
data = pd.read_csv("../data/%s/input_data.txt" % dataset, index_col= 0)
data.index.name = "index"
feature_names = list(data.columns)[0:-1]
label_name = list(data.columns)[-1]

X = data[feature_names]
y = data[label_name]

cat_features = []
flt_features = []
for i in feature_names:
    if type(X.loc[0][i]) == str:
        cat_features.append(i)
    else:
        flt_features.append(i)       
        
cat_X = X[cat_features]
flt_X = X[flt_features]
transformed_X = pd.concat((cat_X, flt_X), axis = 1)

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(cat_X)
trans_X = enc.transform(cat_X)
trans_X = np.concatenate((trans_X.toarray(),np.array(flt_X)),axis=1)

print("Trans Shape:")
print(trans_X.shape)

scaler = MinMaxScaler()
trans_X = scaler.fit_transform(trans_X)


model = IsoForest()
model.fit(trans_X)
scores = model.predict_proba(trans_X)
model.predict(trans_X,y)

begin = time.time()
explainer = shap.Explainer(model.predict_proba, np.array(trans_X))
#anomaly indices
anomaly_indices = np.where(y == 1)
shap_values = explainer(np.array(trans_X[anomaly_indices]))
print(time.time() - begin)


explanations = []
for ad_idx, ad in enumerate(anomaly_indices):
    explain= np.zeros((X.shape[1],))
    anomaly= trans_X[ad]
    total_cat_dim = sum([len(cat_X[i].unique()) for i in list(cat_X.columns)])
    for true_index,i in enumerate(feature_names):
        if i in flt_features:
            idx = flt_features.index(i) + total_cat_dim
            #print(ex_idx,idx,i)
            explain[true_index] = shap_values.values[ad_idx][idx]
        elif i in cat_features:
            #mapped index
            idx = cat_features.index(i)
            #one-hot-encoding of previous mapped features
            prev_sum = sum([len(cat_X[j].unique()) for j in list(cat_X.columns)[0:idx]])
            #current one-hot-encoding
            ohe = anomaly[prev_sum : prev_sum + len(list(cat_X[i].unique()))]
            sp = shap_values.values[ad_idx][prev_sum : prev_sum + len(list(cat_X[i].unique()))]
            explain[true_index] = np.sum(np.multiply(ohe,sp))
    explanations.append(explain)