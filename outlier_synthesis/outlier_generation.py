#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outlier and explanation generation
@author: dingxueying
"""

from abc import abstractmethod, ABC
import argparse
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn.functional import softplus
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from sklearn.mixture import GaussianMixture
from VAE import VAEAnomalyDetection
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import json
import pandas as pd
import seaborn as sns
import sys

def reconstruct_generation(image,cat_dims,max_range, min_range,one_hot_encoder):
    """
    Reconstruct the real-valued array into one-hot-encoding array and original array
    params: image: the real-valued array that needs processing
            cat_dims: the list of categorical features(the numbers of their classes)
            max_range: the max_range of real-valued features
            min_range: the min_range of real-valued features
            one-hot-encoder: sklearn object, trained during preprocessing
            
    returns: (1) vectors with the one-hot-encoding on categorical features
             (2) transformed vectors with de-normalized real-valued features and categorical features
    """
    image_flt = image[:,np.sum(cat_dims):]
    image_flt = np.clip(image_flt, 0, np.max(image_flt))
    image_origin = (image_flt * (max_range - min_range) + min_range).astype(int)
    #first find the maximum among categorical features
    cur_index = 0
    image_cat = []
    for idx,i in enumerate(cat_dims): 
        index = np.argmax(image[:,cur_index:cur_index+i],axis=1).reshape(-1,1)
        #print(index)
        image_cat.append(index)
        zero_index = np.zeros(image[:,cur_index:cur_index+i].shape)
        for row in range(image.shape[0]):
            zero_index[row,index[row]] = 1
        cur_index = cur_index + i
        if idx == 0:
            cate_features = zero_index
        else:
            cate_features = np.hstack((cate_features,zero_index))
        #print(cate_features)
    #inverse transform the one-hot-encoding
    val = one_hot_encoder.inverse_transform(cate_features)
    if not np.array_equal(val, np.hstack(image_cat)):
        print("Wrong reconstruction")
    return np.hstack((cate_features,image_flt)),np.hstack((val,image_origin))


def reconstruction_to_original(reconstruction, cat_dims,max_range,min_range,one_hot_encoder,is_int=False):
    """
    Find the original features from reconstruction
    Denormalize the reconstruction, and inverse the one-hot-encoding
    params: image: the real-valued array that needs processing
            cat_dims: the list of categorical features(the numbers of their classes)
            max_range: the max_range of real-valued features
            min_range: the min_range of real-valued features
            one-hot-encoder: sklearn object, trained during preprocessing
            is_int: True or False, if the original features are discrete, set to True
            
    returns: transformed vectors with de-normalized real-valued features and categorical features    
    """
    image_flt = reconstruction[:,np.sum(cat_dims):]
    image_origin = (image_flt * (max_range - min_range) + min_range)
    if is_int:
        image_origin = image_origin.astype(int)
    cat_features = reconstruction[:,0:np.sum(cat_dims)]
    val = one_hot_encoder.inverse_transform(cat_features)
    return np.hstack((val,image_origin))


def fit_marginal_gmm(normal_data, cat_dims, dim_flt):
    """
    Fit a mariginal GMM models to the normal data (continuous feature only!)
    param: normal_Data
           cat_dims: categorical dimensions
           dim_flt: continuous dimensions
    return: dict: (1) fitted_gaussian, 
                  (2) fitted_min_log_probabiltiy,
                  (3)maximum vals,
                  (4)mininmum vals
    """
    fitted_gaussian = []
    fitted_min_log = []
    val_max_lst = []
    val_min_lst = []
    for dim in range(dim_flt):
        X = normal_data[:,np.sum(cat_dims) + dim].reshape(-1,1)
        N = np.arange(1, 11)
        gaussian_models = [None for i in range(len(N))]
        for i in range(len(N)):
            gaussian_models[i] = GaussianMixture(N[i]).fit(X)
        BIC = [m.bic(X) for m in gaussian_models]
        best_fitted = gaussian_models[np.argmin(BIC)]
        fitted_gaussian.append(best_fitted)
        min_log_prob = np.min(best_fitted.score_samples(X))
        print("Min log prob:", min_log_prob)
        fitted_min_log.append(min_log_prob)
        val_max = np.max(X, axis=0)
        val_min = np.min(X, axis=0)
        val_max_lst.append(val_max)
        val_min_lst.append(val_min)
    return {'fitted_gaussian':fitted_gaussian, 'fitted_min_log': fitted_min_log, 'val_max_lst':val_max_lst,\
                'val_min_lst':val_min_lst}

def multinomial_distribution(normal_data,cat_dims):
    """
    Model one multinomial distribution for all categorical data
    Parameters: normal-data, cat_dim: categorical feature lists
    """
    begin_index = 0
    cat_classifier = []
    for idx,i in enumerate(cat_dims): 
        cur_data = normal_data[:,begin_index:begin_index +i]
        probability = np.sum(cur_data,axis = 0)/ cur_data.shape[0]
        cat_classifier.append(probability)
        begin_index += i
    return cat_classifier


def local_real_value_generation(dim,fitted_gaussian,fitted_min_log):
    """
    Generate real-valued features locally
    Inflate the GMM at index i, with larger covariance
    """
    i = dim
    min_log = fitted_min_log[i]
    best_fitted = fitted_gaussian[i]
    
    log_prob = min_log
    while log_prob >= min_log:
        clf_local = deepcopy(best_fitted)
        clf_local.covariances_ *= 5
        (outlier_local, _) = clf_local.sample(1)
        log_prob = best_fitted.score_samples(outlier_local)
    return outlier_local

            
def global_real_value_generation(dim,fitted_gaussian, fitted_min_log, val_max_lst, val_min_lst):
    """
    Generate real-valued features locally
    Inflate the GMM at index i, by uniformly sampling from extended min range and extended max range
    """
    i = dim
    clf = fitted_gaussian[i]
    min_log = fitted_min_log[i]
    val_max = val_max_lst[i]
    val_min = val_min_lst[i]
    val_range = val_max - val_min

    log_prob = min_log
    outlier_global = 0.0
    while log_prob >= min_log:
        outlier_global = np.random.rand(1,1)
        outlier_global = outlier_global * val_range*3.0 + (val_min - 0.5*val_range) # 10% on each side
        log_prob = clf.score_samples(outlier_global)
    return outlier_global


def categorical_value_generation(dim,cat_dims,categorical_val):
    """
    Categorical features generation
    """
    i = cat_dims[dim]
    prob = categorical_val[dim]
    index= np.argmin(prob)
    val = np.zeros((i,))
    val[index] = 1
    return val,index


#encode the categorical features to numbers with LabelEncoder
def encode_catgorical_column(data, columns):
    encoders = []
    for column in columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        encoders.append(le)
    return encoders


def decode_catgorical_column(data, columns, encoders):
    for i,column in enumerate(columns):
        data[column]= encoders[i].inverse_transform(data[column])
    return data


def generate_anomalies(num_outlier_points,
                       outlier_flt_dimension,
                       outlier_cat_dimension,
                       model,
                       cat_dims,
                       dim_flt,
                       max_range,
                       min_range,
                       one_hot_encoder,
                       threshold,
                       type_,
                       fitted_gaussian,
                       fitted_min_log,
                       val_max_lst,
                       val_min_lst,
                       categorical_val,
                       epsilon = 0.5,
                       group_ = True,
                       group_number =5,
                       pos = True,
                       device = "cuda"
                       ):
    outlier_points_lst = []
    outlier_explanation_lst = []
    count = 0
    while len(outlier_points_lst) < num_outlier_points:
        print(count)
        count += 1
        image = model.generate(1)
        image = image.detach().cpu().numpy()
        reconstruction_,original_ = reconstruct_generation(image,cat_dims,max_range, min_range,one_hot_encoder)
        prob = model.reconstructed_probability(torch.tensor(reconstruction_).float().to(device))
        prob = prob.detach().cpu().numpy()[0]
        if prob < threshold:
            outlier_points = reconstruction_
            outlier_scores = prob
            outlier_explanation = np.zeros((dim_flt,))
            cat_outlier_explanation = np.zeros((cat_dims.shape[0],))
            reconstructed_outlier_point = deepcopy(reconstruction_)
            #generate outlier in flt dimension
            for i in outlier_flt_dimension:
                outlier_point_temp = deepcopy(outlier_points)
                if type_ == "global":
                    outlier = global_real_value_generation(i,fitted_gaussian, fitted_min_log, val_max_lst, val_min_lst)
                elif type_ == "local":
                    outlier = local_real_value_generation(i,fitted_gaussian,fitted_min_log)
                else:
                    print("inappropriate type: ", type_)
                outlier_point_temp[:,i+np.sum(cat_dims)]  = outlier
                new_prob = model.reconstructed_probability(torch.tensor(outlier_point_temp).float().to(device)).item()
                prob_diff = new_prob - prob
                prob_diff = np.max([prob_diff, 0])
                #if lower the probability
                if prob_diff > 0:
                    if pos:
                        if outlier > 0:
                            reconstructed_outlier_point[:,i+np.sum(cat_dims)] = outlier
                    if not pos:
                        reconstructed_outlier_point[:,i+np.sum(cat_dims)] = outlier
                outlier_explanation[i] = prob_diff

            #generateoutlier in cat dimsension
            for i in outlier_cat_dimension:
                outlier_point_temp = deepcopy(outlier_points)
                outlier,outlier_index = categorical_value_generation(i,cat_dims,categorical_val)
                if i == 0:
                    outlier_point_temp[:,0:cat_dims[i]] = outlier
                else:
                    outlier_point_temp[:,np.sum(cat_dims[0:i]):np.sum(cat_dims[0:i+1])] = outlier
                new_prob = model.reconstructed_probability(torch.tensor(outlier_point_temp).float().to(device)).item()
                prob_diff = new_prob - prob
                prob_diff = np.max([prob_diff, 0])
                #if lower the probability
                if prob_diff > 0:
                    if i == 0:
                        reconstructed_outlier_point[:,0:cat_dims[i]] = outlier
                    else:
                        reconstructed_outlier_point[:,np.sum(cat_dims[0:i]):np.sum(cat_dims[0:i+1])] = outlier
                cat_outlier_explanation[i] = prob_diff

            #check reconstructed_outlier_point
            new_all_prob = model.reconstructed_probability(torch.tensor(reconstructed_outlier_point).float().to(device)).item()
            if np.count_nonzero(np.hstack((cat_outlier_explanation,outlier_explanation))) < \
                                                                len(outlier_flt_dimension) + len(outlier_cat_dimension):
                continue
            if new_all_prob > threshold + epsilon * abs(threshold):
                if group_:
                    for i in range(group_number):
                        copied_outlier = deepcopy(reconstructed_outlier_point)
                        copied_outlier[:,np.sum(cat_dims):] += np.random.normal(size=(dim_flt,))* 0.01
                        outlier_points_lst.append(copied_outlier)
                        outlier_explanation_lst.append(np.hstack((cat_outlier_explanation,outlier_explanation)))
                else:
                    outlier_points_lst.append(reconstructed_outlier_point)
                    outlier_explanation_lst.append(np.hstack((cat_outlier_explanation,outlier_explanation)))
    return dict(points = outlier_points_lst, explanations= outlier_explanation_lst)
  
    

def reorder_points(original_outlier_lst,outlier_explanations,list_cat, list_flt,feature_names):
    reordered_lst = []
    reordered_explanations = []
    for origin,exp in zip(original_outlier_lst,outlier_explanations):
        reordered_original = np.zeros(origin.shape)
        reordered_explain = np.zeros(origin.shape)
        for idx,val in enumerate(list_cat):
            reordered_original[:,val] = origin[:,idx]
            reordered_explain[:,val] = exp[idx]
        for idx,val in enumerate(list_flt):
            reordered_original[:,val] = origin[:,len(list_cat)+idx]
            reordered_explain[:,val] = exp[len(list_cat)+idx]
        #reordered_original = np.concatenate((reordered_original,np.ones((1,1))),axis =1)
        reordered_lst.append(reordered_original)
        reordered_explanations.append(reordered_explain)
    reordered_lst = np.array(reordered_lst).astype(int)
    reordered_lst= reordered_lst.squeeze(1)
    print(reordered_lst.shape)
    df = pd.DataFrame(data = reordered_lst, 
                  columns = feature_names)
    return df, reordered_explanations


class Parameters():
    def __init__(self, 
                 json_file_name,
                 ):
        with open(json_file_name, 'r') as openfile:
            json_ = json.load(openfile)
        self.dataset_name = json_["dataset_name"]
        self.input_file = json_["input_file"]
        self.num_normal_points = json_["num_normal_points"]
        self.num_outlier_points = json_["num_outlier_points"]
        self.num_outlier_clusters = json_["num_outlier_clusters"]
        self.epsilon = json_["epsilon"]
        self.type_ = json_["type_"]
        self.device = json_["device"]
        self.output_path = json_["output_path"]
        self.inflated_features = json_["inflated_features"]
        self.lr = json_["lr"]
        self.iteration = json_["iteration"]
        self.batch_size = json_["batch_size"]
        self.hidden_dim = json_["hidden_dim"]
        self.lr = json_["lr"]
        self.batch_size = json_["batch_size"]
        self.iteration = json_["iteration"]
        

def main(parameter_file="vae_parameters.json"):
    parameters = Parameters(parameter_file)
    #specify a GPU model, if not avaialbe
    device = parameters.device
    #load the dataset
    bank_raw = pd.read_csv(parameters.input_file, delimiter = ",",index_col = "index")
    
    print(bank_raw.info())
    
    label_column= "anomaly"
    matrix1 = bank_raw.copy()
    feature_names = list(bank_raw.columns)
    
    print("feature names:", feature_names)
    
    matrix1 = matrix1[feature_names]
    
    num_normal_points = parameters.num_normal_points
    num_outlier_points = parameters.num_outlier_points
    num_outlier_clusters = parameters.num_outlier_clusters
    
    cluster_outlier_features = []
    for i in range(num_outlier_clusters):
        cluster_outlier_features.append(parameters.inflated_features[i])
    
    
    #how far the outliers from the normal points
    epsilon = parameters.epsilon
    type_ = parameters.type_
    lr = parameters.lr
    iteration = parameters.iteration
    batch_size = parameters.batch_size
    hidden_dim = parameters.hidden_dim
    
    #find where the categorical, flatten dimensions are
    list_cat = []
    list_cat_name = []
    list_flt = []
    for i in range(len(matrix1.dtypes)):
        if matrix1.dtypes[i] == int or matrix1.dtypes[i] == float:
            list_flt.append(i)
        else:
            list_cat.append(i)
            list_cat_name.append(matrix1.columns[i])
            
    #map to the original index
    feature_name_mapping = {}
    for i,val in enumerate(list_cat):
        feature_name_mapping[feature_names[val]] = i
    for i,val in enumerate(list_flt):
        feature_name_mapping[feature_names[val]] = i + len(list_cat)
    #feature_name_mapping['y'] = -1
    
    #find the feature mapping of the outlier index:
    outlier_dimension_map = {}
    for idx,i in enumerate(cluster_outlier_features):
        outlier_dimension_map[idx] = [[],[]]
        for outlier_feat in i:
            if feature_name_mapping[outlier_feat] < len(list_cat):
                outlier_dimension_map[idx][1].append(feature_name_mapping[outlier_feat])
            else:
                outlier_dimension_map[idx][0].append(feature_name_mapping[outlier_feat] - len(list_cat))
                
    encoders = encode_catgorical_column(matrix1,list_cat_name)
    #encode_catgorical_column(matrix1, ["y"])
    Data = ((matrix1.values).astype(float))[0:,:]
    data_cat = Data[:,list_cat]
    data_cont = Data[:,list_flt]
    max_range = data_cont.max(axis=0)
    min_range = data_cont.min(axis=0)
    data_std = (data_cont - data_cont.min(axis=0))/ (data_cont.max(axis=0) - data_cont.min(axis=0))
    
    #fit the one_hot_encoding to the categorical features
    one_hot_encoder = preprocessing.OneHotEncoder()
    one_hot_encoder.fit(data_cat)
    one_hot_labels = one_hot_encoder.transform(data_cat).toarray()
    one_hot_cats = one_hot_encoder.categories_
    cat_dims = []
    for i in one_hot_cats:
        cat_dims.append(i.shape[0])
    
    #cat_dims: a list, each index in the list corresponds to how many classes are for one cateorical
    #feature
    cat_dims = np.array(cat_dims)
    
    #data processed to feed into the VAE
    data_decompressed = np.hstack((one_hot_labels,data_std))
    
    #flattened dimensions
    dim_flt = data_decompressed.shape[1] - np.sum(cat_dims)
    
    #fitting the vae now....
    model = VAEAnomalyDetection(data_decompressed[0].shape[0],hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr)
    dloader = DataLoader(data_decompressed,batch_size =batch_size)
    for train_epoch in tqdm(range(iteration)):
        for data in dloader:
            image = data.float()
            image = image.to(device)
            loss = model(image)['loss']
            opt.zero_grad()
            loss.backward()
            opt.step()
            
    #generate NORMAL points
    image = model.generate(int(num_normal_points*2))
    image = image.detach().cpu().numpy()
    
    #reconstructed normal points with categorical features
    reconstruction,original = reconstruct_generation(image,cat_dims,max_range, min_range,one_hot_encoder)
    #calculate the reconstruction probability (anomaly scores)
    prob = model.reconstructed_probability(torch.tensor(reconstruction).float().to(device))
    prob = prob.detach().cpu().numpy()
    prob_order = prob.argsort()
    sorted_prob = prob[prob_order]
    sorted_reconstruction = reconstruction[prob_order]
    sorted_original = original[prob_order][0:num_normal_points]
    
    #use the num_normal_points (should be densely clustered in low-dimensional space)
    generated_normal_points = sorted_reconstruction[0:num_normal_points]
    generated_prob = sorted_prob[0:num_normal_points]
    threshold = np.max(generated_prob)
    print("Normal points shape: ", generated_normal_points.shape)
    print("Normal points threshold:", threshold)
    
    #find the marginal gmms on real-valued features
    gmm_dict = fit_marginal_gmm(generated_normal_points,cat_dims,dim_flt)
    fitted_gaussian = gmm_dict['fitted_gaussian']
    fitted_min_log= gmm_dict['fitted_min_log']
    val_max_lst= gmm_dict['val_max_lst']
    val_min_lst=gmm_dict['val_min_lst']
    
    #find the categorical probabilitys on categorical features'
    categorical_val = multinomial_distribution(generated_normal_points,cat_dims)
       
    #generate outlier points
    outlier_points = []
    outlier_explanations = []
    for i in outlier_dimension_map.keys():
        print('=================================================')
        print('Current i: %d'% i)
        outlier_flt_dimension = outlier_dimension_map[i][0]
        outlier_cat_dimension = outlier_dimension_map[i][1]
        outliers_dict= generate_anomalies(num_outlier_points=num_outlier_points,
                                          outlier_flt_dimension=outlier_flt_dimension,
                                          outlier_cat_dimension=outlier_cat_dimension,
                                          model=model,
                                          cat_dims=cat_dims,
                                          dim_flt=dim_flt,
                                          max_range=max_range,
                                          min_range=min_range,
                                          one_hot_encoder=one_hot_encoder,
                                          threshold=threshold,
                                          type_=type_[i],
                                          fitted_gaussian=fitted_gaussian,
                                          fitted_min_log=fitted_min_log,
                                          val_max_lst=val_max_lst,
                                          val_min_lst=val_min_lst,
                                          categorical_val= categorical_val,
                                          group_number = num_outlier_points,
                                          group_ = True,
                                          epsilon=epsilon,
                                          device=device)
        outlier_points.extend(outliers_dict['points'])
        outlier_explanations.extend(outliers_dict['explanations'])
     
    #transform normal and outlier data points
    original_outlier_lst = []
    for i in outlier_points:
        val = reconstruction_to_original(i, cat_dims,max_range,min_range,one_hot_encoder)
        original_outlier_lst.append(val)
        
    reordered_original = np.zeros((sorted_original.shape[0], sorted_original.shape[1]))
    for idx,val in enumerate(list_cat):
        reordered_original[:,val] = sorted_original[:,idx]
    for idx,val in enumerate(list_flt):
        reordered_original[:,val] = sorted_original[:,len(list_cat)+idx]
    
    nor_df = pd.DataFrame(data =reordered_original, 
                      columns = feature_names)
    out_df, out_explain= reorder_points(original_outlier_lst,outlier_explanations,list_cat,list_flt,feature_names)
    
    feature_to_number_map = {}
    for idx,i in  enumerate(list_cat_name):
        le_name_mapping = dict(zip(encoders[idx].classes_, encoders[idx].transform(encoders[idx].classes_)))
        feature_to_number_map[i] = le_name_mapping
      
    #transform back to original catgorical features
    decode_catgorical_column(nor_df,list_cat_name,encoders)
    decode_catgorical_column(out_df,list_cat_name,encoders)
    
    #add y-label columns
    y = np.array([0] * num_normal_points + [1] * num_outlier_points*num_outlier_clusters )
    y_df = pd.DataFrame(data = y, columns = [label_column])
    
    #save dataframe
    final_df = pd.concat([nor_df,out_df],ignore_index= True)
    final_df = final_df.join(y_df)
    final_df.to_csv(parameters.out_path + '/' + 'generated_synthetic.txt',index=True, sep = ",")
    
    #save explanations
    out_explain = np.array(out_explain)
    out_explain = out_explain.reshape((out_explain.shape[0], out_explain.shape[2]))
    np.savetxt(parameters.out_path + "/" + "explanation_synthetic.txt", out_explain, delimiter = ",")
    
    
if __name__ == '__main__':
    parameter_file = sys.argv[1]
    main(parameter_file)
    