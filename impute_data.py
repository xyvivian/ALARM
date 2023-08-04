#!/usr/bin/env python
# coding: utf-8



import os
import pandas as pd
import sys
import json
import numpy as np



def remove_columns_with_single_value(df):
    column_indices = df.columns
    columns_to_remove = []
    for col in column_indices:
        unique_values = df[col].unique()
        if len(unique_values) == 1:
            columns_to_remove.append(col)
    print(columns_to_remove)
    updated_df = df.drop(columns=columns_to_remove)
    return updated_df, columns_to_remove


def impute_values(df):
    # Loop through all columns in the DataFrame
    for col in df.columns:
        # Check if the column has mixed-type data
        if any(isinstance(value, str) for value in df[col]) and any(isinstance(value, (int, float)) for value in df[col]):
            # Remove rows where the column has mixed-type data
            df = df[~df[col].apply(lambda x: isinstance(x, str))]
    df.fillna(value=0, inplace=True)
    df = df.replace([np.inf], 1e8)
    df = df.replace([-np.inf], -1e8)
    return df

def mask_output_column(df,ones= ["Brute Force"]):
    val =df.iloc[:, -1].unique()
    mask_map = {}
    for i in val:
        print(i.lower())
        if starts_with_substring(i,ones):
            mask_map[i] = 1
        else:
            mask_map[i] = 0
    df.iloc[:, -1] = df.iloc[:, -1].replace(mask_map) 
    values = [0, 1]
    df = df[df.iloc[:,-1].isin(values)]
    print("labels: ", df.iloc[:, -1].unique())
    return df

def starts_with_substring(s, substr_list):
    s = s.lower()
    for substr in substr_list:
        if s.startswith(substr.lower()):
            return True
    return False

#flatten a list
def flatten_list(lst):
    flattened_str = '_'.join(str(item) for item in lst)
    return flattened_str

def main(masked):
    # Define the directory containing the files you want to read
    directory = "data"
    # Define the output file name and location
    output_file = "data/processed/preprocessed_%s.csv" % flatten_list(masked)
    # Iterate through the files in the directory
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):  # Check if the file is a pd file
            filepath = os.path.join(directory, filename)
            data = pd.read_csv(filepath)  # Read the data into a DataFrame
            print("data loaded with shape: ", data.shape)
            if filename == "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv": #truncate the data
                data = data.iloc[:, 4:]
            ret_df = data.drop(columns=["Timestamp"])
            ret_df = impute_values(ret_df)    #impute non-int/float values in mixed-type arr
            if count == 0:
                ret_df, columns_to_remove = remove_columns_with_single_value(ret_df) #remove cols that only have one value!
            else:
                ret_df = ret_df.drop(columns=columns_to_remove)
            count += 1
            ret_df = mask_output_column(ret_df,masked)
            print("processed columns: ", len(ret_df.columns))
            
            # Write the new DataFrame to the existing CSV file
            ret_df.to_csv(output_file, mode="a", header=False, index=False)
            print("imputed data saved")

    #Count the number of ones and zeros in the resulting output file
    # Load the CSV file without column names
    df = pd.read_csv(output_file, header=None)
    print(df.head(100))
    # Count the occurrences of each value in the 'column_name' column
    value_counts = df.iloc[:, -1].value_counts()
    # Get the count of 1s and 0s
    count_1 = value_counts[1]
    count_0 = value_counts[0]
    print("Ratio of 1s:", count_1 / (count_1 + count_0))
    print("Ratio of 0s:", count_0 / (count_1 + count_0))
    print("Total number of columns: ", len(df.columns))
    print("Has Nan value : {}".format(df.isna().any().any()))
    print("Has Null value : {}".format(df.isnull().any().any()))
    print("Has Inf value : {}".format(np.isinf(df.values).any().any()))



            
if __name__ == "__main__":
    dat_file = sys.argv[1]
    with open("json/" + dat_file + ".json", "r") as f:
        masked = json.load(f)['attack_type']
    main(masked)
