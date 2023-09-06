import sys
import json
from functools import reduce
import os

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def flatten_list(lst):
    flattened_str = '_'.join(str(item) for item in lst)
    return flattened_str

filename = sys.argv[1]
with open("json/"+filename, "r") as f:
    data = json.load(f)
   
#Sparks parameters
repartition = False
numPartitions = 128
nthreads_fit = 128
nthreads_score = 128
samplerate = 0.01
projection = True

nchains = data['nchains']
depth = data['depth']
projdim = data['projdim']
attack_type = flatten_list(data['attack_type'])
input_file = "data/processed/preprocessed_%s.csv" % attack_type

r = data['r']
w = data['w']

create_directory_if_not_exists("results/%s" % attack_type)
output_file = "results/%s/output_score_" % attack_type + str(nchains) + "_" + str(depth) + "_" + str(r) + "_" + str(w) + ".txt"
output_pickle_file = "results/%s/sketches_" % attack_type + str(nchains) + "_" + str(depth) + "_" + str(r) + "_" + str(w) + ".pkl"
output_mask_file = "results/%s/mask_" % attack_type + str(nchains) + "_" + str(depth) + "_" + str(r) + "_" + str(w) + ".npy"
output_r_file = "results/%s/r_" % attack_type + str(nchains) + "_" + str(depth) + "_" + str(r) + "_" + str(w) + ".npy"
output_result_file = "results/%s/output_result_" % attack_type + str(nchains) + "_" + str(depth) + "_" + str(r) + "_" + str(w)
output_feature_file = "results/%s/output_features_" % attack_type + str(nchains) + "_" + str(depth) + "_" + str(r) + "_" + str(w) + ".npy"
output_fs_file = "results/%s/chains_fs_" % attack_type + str(nchains) + "_" + str(depth) + "_" + str(r) + "_" + str(w) + ".npy"


# when input data contains no strings, can set mutable_proj_matrix=False for faster projection
mutable_proj_matrix = False

sys.path.append("/opt/packages/spark/latest/python/lib/py4j-0.10.9-src.zip")
sys.path.append("/opt/packages/spark/latest/python/")
sys.path.append("/opt/packages/spark/latest/python/pyspark")

from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext

spark = SparkSession.builder\
    .config("spark.driver.memory", "128g")\
    .config("spark.driver.maxResultSize", "128g")\
    .config("spark.memory.fraction", "0.6")\
    .config("spark.executor.instances", "64")\
    .config("spark.executor.memory", "128g")\
    .config("spark.executor.cores", "256")\
    .config("spark.network.timeout", "300s") \
    .getOrCreate()
sc = spark.sparkContext

print(f'num executors: {sc.getConf().get("spark.executor.instances")}')
print(f'executor cores: {sc.getConf().get("spark.executor.cores")}')
print(f'executor memory: {sc.getConf().get("spark.executor.memory")}')
print(f'executor pyspark memory: {sc.getConf().get("spark.executor.pyspark.memory")}')
print(f'driver memory: {sc.getConf().get("spark.driver.memory")}')




import time
import math
import numpy as np
import random
import mmh3
import pandas as pd
import tqdm
import pickle
import matplotlib.pyplot as plt
from pyspark.sql.functions import col

from sklearn.metrics import roc_auc_score, average_precision_score
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.functions import monotonically_increasing_id, col
from pyspark.sql.functions import col, isnan

start = time.time()
myDF = spark.read.options(inferSchema='True',header='False').csv(input_file)
myDF = myDF.na.fill('').na.fill(0)
# replace rows with "inf" values with very large numbers
#myDF = myDF.replace("inf", 1e8)  #UPDATED

end = time.time()
count = myDF.count()
print("Read time = %.3fs" % (end - start))
print(myDF.rdd.getNumPartitions())
print((count, len(myDF.columns)))

#assign original indexing 
myDF = myDF.withColumn('index', monotonically_increasing_id())
columns = ['index'] + myDF.columns[0:-1]
myDF = myDF.select(*columns)

if repartition:
    myDF = myDF.repartition(numPartitions)
    print("Repartition:", myDF.rdd.getNumPartitions())
print((myDF.count(), len(myDF.columns)))

all_columns = myDF.columns[1:-1]


from pyspark.sql.functions import max as sparkMax
from pyspark.sql.functions import min as sparkMin

def min_max_normalize(df):
    feature_names = all_columns
    # getting the max and min for all columns
    max_df = df.select(*[sparkMax(col(c)).alias(c) for c in feature_names])
    all_max = max_df.toPandas().to_numpy()[0]
    min_df = df.select(*[sparkMin(col(c)).alias(c) for c in feature_names])
    all_min = min_df.toPandas().to_numpy()[0]

    for i in range(len(feature_names)):
        if ((df.dtypes[i][1]) == 'string'):
            continue
        # Iterate over all columns and cast to float
        df = df.withColumn(feature_names[i], col(feature_names[i]).cast("float"))  #NE
        min_val = float(all_min[i])
        max_val = float(all_max[i])
        df = df.withColumn(feature_names[i],(col(feature_names[i])- min_val) / (max_val - min_val + 1e-9))   #prevent overflow
    return df

class StreamhashProjection:

    def __init__(self, n_components, feature_names=None, density=1/3.0, random_state=None, mutable=False, sc=None):
        self.keys = np.arange(0, n_components, 1)
        self.constant = np.sqrt(1./density)/np.sqrt(n_components)
        self.density = density
        self.n_components = n_components
        self.mutable=mutable
        random.seed(random_state)
        
        input_dim = len(feature_names)
        self.R = sc.broadcast(np.array([[self._hash_string(k, f) # shape: (len(self.keys), input_dim)
                       for f in feature_names]
                       for k in self.keys]))
        
    def get_R(self):
        return self.R.value

    def fit_transform(self, X, feature_names=None):    
        if self.mutable:
            #nsamples = 1      # X.shape[0]
            ndim = len(X)     # X.shape[1]
            if feature_names is None:
                feature_names = [str(i) for i in range(ndim)]
            types = [type(X[i]) == str for i in range(ndim)] #!!! [0][i] not [i]
            feature_names = [feature_names[i]+'.'+X[i] if types[i] else feature_names[i] for i in range(ndim)]
            X = [1 if types[i] else X[i] for i in range(ndim)]

            R = self.R.value
            for i in range(ndim):
                if types[i]:
                    f = feature_names[i]
                    R[:,i] = np.array([self._hash_string(k, f) for k in self.keys])
            Y = np.dot(X, R.T)
        else:
            Y = np.dot(X, self.R.value.T)
        return Y
    
    def _hash_string(self, k, s):
        hash_value = int(mmh3.hash(s, signed=False, seed=k))/(2.0**32-1)
        den = self.density
        if hash_value <= den/2.0:
            return 1
        elif hash_value <= den:
            return 1
        else:
            return 0

def projectDF(df, projdim, sc=None):
    feature_names = all_columns
    projector = StreamhashProjection(n_components=projdim, 
                                     feature_names=feature_names,
                                     density=1/3.0, 
                                     random_state=42, 
                                     mutable=mutable_proj_matrix,
                                     sc=sc)
    R = projector.get_R()
    print(R)
    projectedDF = df.rdd.map(lambda x: [x[0]] + projector.fit_transform(list(x[1:-1]),feature_names).tolist() + [x[-1]]).toDF()  
    return projectedDF,R

def feature_range(df):
    features = df.columns[1:-1]
    deltamax = np.zeros(len(features), dtype=float)  
    for f in range(len(features)):
        deltamax[f] = (df.agg({features[f]: "max"}).collect()[0][0] + 1e-9 - df.agg({features[f]: "min"}).collect()[0][0])/2
    return deltamax

np.random.seed(42)
from array import array

class CMS:
    def __init__(self, r, w, mask = None):
        self.r = r
        self.w = w
    
        upper_bound = 2147483647
        step = upper_bound / (r-1)
        manyranges = [(i*step, step*(i+1)-1) for i in range(r-1)]
        if mask is not None:
          self.mask = mask
        else:
          self.mask = array('L', (np.random.randint(low, high) for low, high in manyranges))
    
    def get_CMS_mask(self):
        return self.mask

    def findAllRowCols(self, myL):          
        h = hash(str(myL)) % self.w

        result = []
        result.append( (1, h) )
        row=2
        for m in self.mask:
            result.append( (row, (h ^ m) % self.w ))
            row+=1
        return result

    def allCols(self, X):
        h = hash(str(X)) % self.w

        result = []          
        result.append( ((1, h), 1) )
        row=2
        for m in self.mask:
            result.append( ((row, (h ^ m) % self.w), 1) )
            row+=1
        return result

cms = CMS(r=r, w=w)

class Chain:
  
    def __init__(self, depth, deltamax):
        k = len(deltamax)
        self.deltamax = deltamax # feature ranges
        self.depth = depth
        self.fs = [np.random.randint(0, k) for d in range(depth)]
        ### self.cmsketches = [{}] * depth #self.cmsketches = [None] * depth
        self.shift = np.random.rand(k) * deltamax

    def get_init(self):
        return self.fs, self.shift

    # input X is a **single** point 
    def fit(self, X, verbose=False): #all depths
        
        # initialize cmsketch tables
        prebin = np.zeros(len(X), dtype=float)
        depthcount = np.zeros(len(self.deltamax), dtype=np.int32)
        
        ls = [None] * depth
        for d in range(self.depth):
          f = self.fs[d] #split feature at depth d
          depthcount[f] += 1

          if depthcount[f] == 1:
              prebin[f] = (X[f] + self.shift[f])/self.deltamax[f] 
          else:
              prebin[f] = 2.0*prebin[f] - self.shift[f]/self.deltamax[f]
          
          
          ls[d] = tuple(np.floor(prebin).astype(np.int32))
          #if not l in cmsketch:
          #    cmsketch[l] = 0
          #cmsketch[l] += 1

        #self.cmsketches[depth] = cmsketch

        return ls #<-- contains the integer bind-id vector at each level for the *single* X

    def bincount_score(self, X, cmsketches):
        # calculate the score at every depth
        scores = np.zeros(self.depth)
        scores_2 = np.zeros(self.depth)
        prebin = np.zeros(len(X), dtype=float)
        depthcount = np.zeros(len(self.deltamax), dtype=np.int32)
  
        for d in range(self.depth):
            f = self.fs[d] 
            depthcount[f] += 1
            if depthcount[f] == 1:
                prebin[f] = (X[f] + self.shift[f])/self.deltamax[f]
            else:
                prebin[f] = 2.0*prebin[f] - self.shift[f]/self.deltamax[f]

            cmsketch = cmsketches[d]
            l = tuple(np.floor(prebin).astype(np.int32))

            rowcols = cms.findAllRowCols(l)
            counts = [cmsketch.get(rowcol, 0) for rowcol in rowcols]
            scores[d] = min(counts)
        depths = np.array([d for d in range(1, self.depth+1)])
        scores = np.log2(1.0 + scores) + depths 
        return np.min(scores)

from multiprocessing.pool import ThreadPool
from functools import reduce
from operator import add

class Chains:
    def __init__(self, deltamax, nchains=1, depth=2, nthreads_fit=1, nthreads_score=128, samplerate=0.1): #k=3 , seed=42
        self.nchains = nchains
        self.depth = depth
        self.depth = depth
        self.nthreads_fit = nthreads_fit
        self.nthreads_score = nthreads_score
        self.samplerate = samplerate
        self.deltamax = deltamax
        self.chains_cmsketches = [None] * nchains
        self.chains = [None] * nchains
        #self.projector = StreamhashProjection(n_components=k, density=1/3.0, random_state=seed)  # StreamhashProjection(n_components=k, density=1/3.0, random_state=seed)

    def get_all_init(self):
        ret_variables = []
        for idx in range(self.nchains):
            chain = self.chains[idx]
            fs,shift = chain.get_init()
            cms_sketch = self.chains_cmsketches[idx]
            ret_variables.append({"fs":fs,"shift":shift,"sketch":cms_sketch})
        return ret_variables

    def get_all_fs(self):
        ret_variables = []
        for idx in range(self.nchains):
            cms_sketch  = self.chains_cmsketches[idx]
            ret_variables.append(cms_sketch)
        return np.array(ret_variables)


    def fitparallel(self, allX):   
        pool = ThreadPool(self.nthreads_fit)

        def fitone(cindex):     
            # create i'th Chain
            c = Chain(depth=self.depth, deltamax=self.deltamax)
            self.chains[cindex] = c

            binvecsRDD = allX.sample(False,self.samplerate, cindex*100).map(lambda x: c.fit(x[0])) # <-- inject random sampling

            cmsketches = [{}] * self.depth
            for d in range(self.depth):
                cmsketches[d] = binvecsRDD.flatMap(lambda x: cms.allCols(x[d])).reduceByKey(lambda x,y:x+y).collectAsMap()
            self.chains_cmsketches[cindex] = cmsketches
            
            return [cindex]
        
        parameters = list(range(self.nchains))
        pool.map(lambda chainindex: fitone(chainindex), parameters)
        return self

    def fit(self, allX):     
        for i in tqdm.tqdm(range(self.nchains), desc='Fitting...'):
            # create i'th Chain
            c = Chain(depth=self.depth, deltamax=self.deltamax)
            self.chains[cindex] = c

            binvecsRDD = allX.rdd.sample(False,self.samplerate,i*100).map(lambda x: c.fit(x)) # <-- inject random sampling

            cmsketches = [{}] * self.depth
            for d in range(self.depth):
                cmsketches[d] = binvecsRDD.map(lambda x: (x[d], 1)).reduceByKey(lambda x,y:x+y).collectAsMap()
            self.chains_cmsketches[cindex] = cmsketches
        
        return self

    def scoreparallel(self, allX):
        time_a = time.time()
        pool = ThreadPool(self.nthreads_score)

        def scoreone(cindex):     
            chain = self.chains[cindex]
            scores_rdd = allX.map(lambda x:(chain.bincount_score(x[0], self.chains_cmsketches[cindex]), x[1]))
            return scores_rdd.zipWithIndex().map(lambda x:(x[1], x[0])) # (index, (score, label))
        
        parameters = list(range(self.nchains))
        all_score_rdds = pool.map(lambda chainindex: scoreone(chainindex), parameters)
        assert(len(all_score_rdds) == self.nchains)
        time_b = time.time()
        print("Score time = ", time_b-time_a)
        
        sum_rdd = all_score_rdds[0]
        for i in range(1, self.nchains):
            score_rdd = all_score_rdds[i]
            sum_rdd = sum_rdd + score_rdd
            
        sum_rdd = sum_rdd.reduceByKey(lambda x,y:(x[0]+y[0], x[1]))
        time_c = time.time()
        print("Combine time = ", time_c-time_b)
        return sum_rdd.map(lambda x:(float(x[1][1][0]),float(-x[1][0]/self.nchains), float(x[1][1][1]))) # (score, label)

def run_xstream_all():
    
    all_outputs = []
    min_max = False
    if min_max:
        print("Min-max normalize")
        all_outputs.append("Min-max normalize")
        start = time.time()
        normDF = min_max_normalize(myDF) #100+ columns
        normDF.show()
        end = time.time()
        print("Time = %.3fs" % (end-start))
        all_outputs.append("Time = %.3fs" % (end-start))
    else:
        normDF = myDF

    print("saving masks")
    mask = cms.get_CMS_mask()
    np.save(output_mask_file,mask)
    
    print("Projection")
    all_outputs.append("Projection")
    start = time.time()
    if projection:
        pDF,R = projectDF(normDF, projdim, sc=sc)
        np.save(output_r_file,R)
    else:
        for column in normDF.columns:
            normDFdf = normDF.withColumn(column, col(column).cast("float"))
        pDF = normDF
    deltamax = feature_range(pDF)
    np.save(output_feature_file, deltamax)

    # convert into pair rdd
    pDF_pair = pDF.rdd.map(lambda x: (list(x[1:-1]), [x[0],x[-1]]))
    pDF_pair.cache()
    end = time.time()
    print("Time = %.3fs" % (end-start))
    all_outputs.append("Time = %.3fs" % (end-start))
    
    print("Fitting")
    all_outputs.append("Fitting")
    start = time.time()
    

    cf = Chains(deltamax, nchains=nchains, depth=depth, nthreads_fit=nthreads_fit, nthreads_score=nthreads_score, samplerate=samplerate)
    cf = cf.fitparallel(pDF_pair)
    dump_file = output_pickle_file
    with open(dump_file, "wb") as f:
        pickle.dump(cf.get_all_init(), f)
    
    end = time.time()
    print("Time = %.3fs" % (end-start))
    all_outputs.append("Time = %.3fs" % (end-start))
    
    print("Scoring")
    all_outputs.append("Scoring")
    start = time.time()
    anomalyScoresRDD = cf.scoreparallel(pDF_pair)

    #save the fs chains
    fs = cf.get_all_fs()
    np.save(output_fs_file, fs)
    
    # Extract the second and third columns
    score_rdd = anomalyScoresRDD.map(lambda row: (row[1], row[2]))

    metrics = BinaryClassificationMetrics(score_rdd)
    auc = metrics.areaUnderROC
    ap = metrics.areaUnderPR

    end = time.time()
    print("Time = %.3fs" % (end-start))
    all_outputs.append("Time = %.3fs" % (end-start))

    print("xstream: AUC =", auc)
    all_outputs.append("xstream: AUC = %.5f" % auc)

    print("xstream: AP =", ap)
    all_outputs.append("xstream: AP = %.5f" % ap)
    
    # Sort the anomalyscores by position = 1
    score_df = spark.createDataFrame(anomalyScoresRDD)
    sorted_data = score_df.orderBy(score_df._2.asc())
    
    with open(output_file, 'w') as f:
        for s in all_outputs:
            print(s, file=f)
            
    # Save the RDD
    sorted_data.coalesce(1).write.csv(output_result_file)
           



run_xstream_all()

