import sys
sys.path.append("DIFFI")

import os
import numpy as np
import pickle as pkl 
import time
import matplotlib.pyplot as plt 
%matplotlib inline
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import shuffle
import shap
import interpretability_module as interp
from utils import *
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

#prepare for data
dataset = "pkdd1998"
data = pd.read_csv("../data/%s/generated_synthetic.txt" % dataset, index_col= 0)
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

#min-max-normlize the X
scaler = MinMaxScaler()
trans_X = scaler.fit_transform(trans_X)

#train isolation forest
iforest = IsolationForest(n_estimators= 100, max_samples=256, contamination=0.1, bootstrap=False)
iforest.fit(trans_X)
y_tr_pred = np.array(iforest.decision_function(trans_X) < 0).astype('int')
f1 = f1_score(y, y_tr_pred)
print('F1 score (on training data): {}'.format(round(f1, 3)))
auc = roc_auc_score(y, y_tr_pred)
print('AUROC Score:', auc)

# Local-DIFFI
diffi_te, ord_idx_diffi_te, exec_time_diffi_te = local_diffi_batch(iforest, np.array(trans_X)[np.where(y == 1)])
# the local diffi is given as the following:
diffi_te.shape

#anomaly indices
anomaly_indices = np.where(y == 1)

#convert into true explanations
explanations = []
for ad_idx,ad in enumerate(anomaly_indices):
    explain= np.zeros((X.shape[1],))
    anomaly= trans_X[ad]
    total_cat_dim = sum([len(cat_X[i].unique()) for i in list(cat_X.columns)])
    for true_index,i in enumerate(feature_names):
        if i in flt_features:
            idx = flt_features.index(i) + total_cat_dim
            #print(ex_idx,idx,i)
            explain[true_index] = diffi_te[ad_idx][idx]
        elif i in cat_features:
            #mapped index
            idx = cat_features.index(i)
            #one-hot-encoding of previous mapped features
            prev_sum = sum([len(cat_X[j].unique()) for j in list(cat_X.columns)[0:idx]])
            #current one-hot-encoding
            ohe = anomaly[prev_sum : prev_sum + len(list(cat_X[i].unique()))]
            sp = diffi_te[ad_idx][prev_sum : prev_sum + len(list(cat_X[i].unique()))]
            explain[true_index] = np.sum(np.multiply(ohe,sp))
    explanations.append(explain)
