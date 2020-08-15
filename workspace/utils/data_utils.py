import os
import sys
import time
import numpy as np

DATA_PATH = "data/"
# To load dataset for training
def load_dataset(stock_env, p1, p2, use_filter_data, filter_by_year):
    train_head_name = stock_env + "_train_data" + "_p1" + str(p1) + "_p2" + str(p2)
    if(use_filter_data):
        train_norm_data = np.load(DATA_PATH + train_head_name + "_normalized_filtered_fyear" + str(filter_by_year) + ".npy")
        train_raw_data = np.load(DATA_PATH + train_head_name + "_raw_filtered_fyear" + str(filter_by_year) + ".npy")
    else:
        train_norm_data = np.load(DATA_PATH + train_head_name + "_normalized.npy")
        train_raw_data = np.load(DATA_PATH + train_head_name + "_raw.npy")
    test_head_name = stock_env + "_test_data" + "_p1" + str(p1) + "_p2" + str(p2)
    if(use_filter_data):
        test_norm_data = np.load(DATA_PATH + test_head_name + "_normalized_filtered_fyear" + str(filter_by_year) + ".npy")
        test_raw_data = np.load(DATA_PATH + test_head_name + "_raw_filtered_fyear" + str(filter_by_year) + ".npy")
    else:
        test_norm_data = np.load(DATA_PATH + test_head_name + "_normalized.npy")
        test_raw_data = np.load(DATA_PATH + test_head_name + "_raw.npy")
    return (train_head_name, test_head_name),  train_norm_data, train_raw_data, test_norm_data, test_raw_data
