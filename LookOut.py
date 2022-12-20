import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import copy
from scipy.interpolate import interp1d
import time
import seaborn as sns
from datetime import date
import os
import sys
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
   
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity
from collections import Counter
from itertools import combinations
from matplotlib.ticker import FormatStrFormatter
from sklearn.preprocessing import minmax_scale


sys.path.insert(0, "xstreamPython")
from Chains import Chains

X = np.loadtxt("data/data/drive_data_with_labels.txt",delimiter = ",")[:,1:-1]
y = np.loadtxt("data/data/drive_data_with_labels.txt",delimiter = ",")[:,-1]
X_index = np.loadtxt("data/data/drive_data_with_labels.txt",delimiter = ",")[:,0]


outliers = np.where(y == 1)[0]

class XStream():
    def __init__(self,k=50,nchains=50,depth=10, **kwargs):
        # initialize
        self.xStream = None
        self.k = k
        self.nchains = nchains
        self.depth = depth
        self.initialize_xStream(**kwargs)


    def initialize_xStream(self, seed=0, **kwargs):
        self.xStream = Chains(k=self.k, nchains=self.nchains, depth=self.depth, **kwargs)

    def fit(self, train_X):
        print("Starting training...")
        start_time = time.time()
        self.xStream.fit(train_X.astype(np.float32))
        end_time = time.time() - start_time
        return end_time

    def predict_proba(self,test_X):
        scores = (-1.0) * self.xStream.score(test_X.astype(np.float32))  # compute anomaly score
        return scores
    
    def predict(self,test_X,test_y):
        print("Starting prediction...")
        scores = (-1.0) * self.xStream.score(test_X.astype(np.float32))  # compute anomaly score
        auc = roc_auc_score(test_y, scores.flatten())
        print("AUCROC: %.4f" % auc)
        return scores

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

shap_values = np.loadtxt("data/data/drive_feature_inference_results.projection.50.average.top100.result.txt",delimiter = ",")

# Get scatter plot outlier scores and figure
def get_scores(original_data, feature_X, feature_Y, make_plot=True, get_log = False, top_k = 100):
    data = original_data[:,(feature_X, feature_Y)]
    #you need to feed a subset of features, retraining is required
    new_model = XStream()
    new_model.fit(data)
    print("We acquire the anomaly scores")
    
    scores = new_model.predict_proba(data)
    scores = minmax_scale(scores)
    ids = list(range(data.shape[0]))
    tuples = [(ids[i], scores[i]) for i in range(0, len(ids))]
    scores = sorted( tuples, key = lambda x: x[1], reverse = True )
    scores = scores[0:top_k]
    return scores

original_data = np.loadtxt("data/data/drive_data_with_labels.txt",delimiter = ",")
data_index = original_data[:,0].tolist()

val_original = np.loadtxt("data/data/drive_feature_inference_results.projection.50.average.top100.result.txt",delimiter = ",")
## first axis is the index of the data, we dont need it
explanation_value = val_original[:,1:]
explanation_index = val_original[:,0]

# perform MDS
dist_euclid = euclidean_distances(explanation_value)
mds = MDS(dissimilarity='precomputed', random_state=0)
# Get the embeddings
data_transformed = mds.fit_transform(dist_euclid)

data_MDS = pd.DataFrame(data_transformed, columns= ['x','y'])

sil = []
label_lst = []
for i in range(2,10):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(data_transformed)
    labels = kmeans.labels_
    label_lst.append(labels)
    sil.append(silhouette_score(data_transformed, labels, metric = 'euclidean'))
    col_name =  str(i) + ' clusters'
    data_MDS[col_name]= labels
    
optimal_k = np.argmax(sil) + 2

cluster_data = data_MDS.set_index(explanation_index)

df = cluster_data.reset_index()
df["index"] = df["index"].astype(int)
clusters = ["2 clusters","3 clusters","4 clusters","5 clusters","6 clusters","7 clusters","8 clusters","9 clusters"]
cluster_idx = {}
for cluster in clusters:
    for c in df[cluster].unique():
        cluster_idx[cluster, c] = (list(zip(df[df[cluster] == c].index, df[df[cluster] == c]["index"].values)))   

cluster_features = []

for cluster in cluster_idx.values():
    idx, value = zip(*cluster)
    features_list = np.argsort(np.mean(shap_values[idx,1:],axis=0))[::-1][0:10]
    cluster_features.append(features_list)


def findPairs(n): 
    return list(Counter(combinations(n, 2)))

cluster_feature_pairs = [] 

for features_list in cluster_features:
    cluster_feature_pairs.append(findPairs(features_list))

clusters_rank_plot = []

for feature_pairs in cluster_feature_pairs:
    cluster_scores = []
    for pair in feature_pairs:
        cluster_scores.append(get_scores(X, pair[0],pair[1], top_k=100))
    clusters_rank_plot.append(cluster_scores)

np.save('xstream.npy', np.array(clusters_rank_plot), allow_pickle=True)

df_xstreams = np.load('xstream.npy', allow_pickle=True)

def assign_scores(cluster_rank):
    plot_best = {}
    for plot_n,plot in enumerate(cluster_rank):
        for outlier in plot:
            idx = outlier[0]
            score = outlier[1]
            if idx in outliers: 
                if idx not in plot_best.keys():
                    plot_best[idx]= (plot_n,score)
                elif plot_best[idx][1] < score:
                    plot_best[idx] = (plot_n,score)
    return plot_best

cluster_plot_scores = []

for cluster_rank in df_xstreams:
    plot_scores = {}
    for key,value in assign_scores(cluster_rank).items():
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

    cluster_plot_scores.append((cluster_rank,sorted_plots,best_graphs))

def LookOut(budget,cluster_rank_plot,sorted_plots,best_graphs):
    budget_best_graphs = best_graphs[0:budget]
    budget_plots = []
    for plot,score in sorted_plots:
        if budget > 0:
            budget_plots.append(cluster_rank_plot[plot])
            budget-=1
        else: break
    

    plot_max = {}
    for key,value in assign_scores(budget_plots).items():
        anomaly_id = key
        plot_id = sorted_plots[value[0]][0]
        score = value[1]
        if plot_id not in plot_max.keys():
            plot_max[plot_id] = [score,[anomaly_id]]
        else:
            plot_max[plot_id][0] += score
            plot_max[plot_id][1] += [anomaly_id]
    return plot_max,budget_best_graphs


def plot(budget,cluster_rank_plot,sorted_plots,best_graphs,cluster):
    lo = LookOut(budget,cluster_rank_plot,sorted_plots,best_graphs)
    figure_list = []
    for plot in lo[1]:
        score = lo[0][plot][1]
        x,y = cluster_feature_pairs[cluster][plot]
        dfx = pd.DataFrame(X_index)
        dfx["color"] = [0 if i not in outliers else 1 if i in score else 2 for i in dfx[0]]
        newPal = {0 :'black' , 1 :'red', 2 :'cyan'}
        fig = plt.figure(figsize = (4,3),dpi = 400)
        ax = fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        features = np.load("data/data/dataset_features.npy",allow_pickle=True)
        plt.xlabel(features[x])
        plt.ylabel(features[y])
        plt.scatter(X[:,(x)],X[:,(y)],c=dfx["color"].map(newPal),edgecolor='black',linewidth=0.3,alpha=0.5)
        figure_list.append(fig)
    return figure_list

clusters = [2,2,3,3,3,4,4,4,4,5,5,5,5,5,6,6,6,6,6,6,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9]
sub_cluster = [1,2,1,2,3,1,2,3,4,1,2,3,4,5,1,2,3,4,5,6,1,2,3,4,5,6,7,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,9]

today = date.today()
os.mkdir("assets/"+str(today))

for x,y,z in zip(enumerate(cluster_plot_scores),clusters,sub_cluster):
    idx,(a,b,c) = x
    cluster_id = y
    sub_cluster_id = z
    for budget in range(5):
        figures = plot(budget+1,a,b,c,idx)
        for i,figure in enumerate(figures):
            fname = "assets/"+str(today)+"/{0}-{1}-{2}-{3}-{4}.png".format("lookout",cluster_id,sub_cluster_id,budget+1, i+1)
            figure.savefig(fname,bbox_inches = 'tight')
            plt.close(figure)
