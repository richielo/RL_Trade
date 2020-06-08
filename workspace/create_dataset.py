"""
This script looks for json data in the data directory to create training and testing set based on the 12 features
"""
import os
import sys
import argparse
import numpy as np
import json
import pandas as pd
from feature_extraction_helpers import *

DATA_PATH = "data/"
FILE_EXT = ".json"

def extract_timestamp(item):
	try:
		return int(item['t'])
	except KeyError:
		return 0

def extract_features(file_path, period_1, period_2):
	with open(file_path) as data_file:
		data = json.load(data_file)
		result_json_list = data['results']
		result_dataframe = pd.DataFrame.from_records(result_json_list)

		# Remove All zeroes rows
		all_zeros_rows = result_dataframe[(result_dataframe['o'] ==0) | (result_dataframe['c'] == 0) | (result_dataframe['h'] == 0) | ((result_dataframe['l'] == 0))].index
		result_dataframe.drop(all_zeros_rows, inplace = True)
		result_json_list = json.loads(result_dataframe.to_json(orient = 'records'))

		# Sort according to timestamp
		result_json_list.sort(key = extract_timestamp)

		# Compute technical indicators
		roc_list = ROC(result_json_list, period_1)
		atr_list = ATR(result_json_list, period_1)
		ma_list = MA(result_json_list, period_1)
		ema_list = EMA(result_json_list, period_1)
		macd_list = MACD(result_json_list, period_1, period_2)

		# Extract market variables
		o_list, c_list, h_list, l_list, v_list = Market_Variables(result_json_list)

		# Data size check
		assert len(roc_list) == len(atr_list) == len(ma_list) == len(ema_list) == len(macd_list) == len(o_list) == len(c_list) == len(h_list) == len(l_list) == len(v_list) #== len(t_list)

		features_array = np.array([roc_list, atr_list, ma_list, ema_list, macd_list, o_list, c_list, h_list, l_list, v_list]).T
		return features_array


def normalize(train_set, test_set):
	mean_vec = np.mean(train_set, axis = 0)
	sd_vec = np.std(train_set, axis = 0)
	norm_train_set = (train_set - mean_vec) / sd_vec
	norm_test_set = (test_set - mean_vec) / sd_vec
	return norm_train_set, norm_test_set

def create_dataset(file_path, split_ratio, period_1, period_2):
	if(file_path is None):
		# Go through the entire directory
		for file in os.listdir(DATA_PATH):
			if(file.endswith(FILE_EXT)):
				stock_name = file.split('_')[1]
				file_path = os.path.join(DATA_PATH, file)
				# Extract features
				features_array = extract_features(file_path, period_1, period_2)

				# Train/Test split
				train_size = int(features_array.shape[0] * split_ratio)
				test_size = features_array.shape[0] - train_size
				train_set = features_array[: train_size]
				test_set = features_array[train_size :]
				assert (train_set.shape[0] + test_set.shape[0]) == features_array.shape[0]

				# Normalize dataset
				norm_train_set, norm_test_set = normalize(train_set, test_set)
				assert norm_train_set.shape == train_set.shape
				assert norm_test_set.shape == test_set.shape

				# Save file
				np.save(DATA_PATH + stock_name + "_train_data" + "_p1" + str(period_1) + "_p2" + str(period_2) + ".npy", norm_train_set)
				np.save(DATA_PATH + stock_name + "_test_data" + "_p1" + str(period_1) + "_p2" + str(period_2) +".npy", norm_test_set)
	else:
		# Process individual file
		# Assume the file in the DATA_PATH directory
		stock_name = file_path.split('/')[1].split('_')[1]
		# Extract features
		features_array = extract_features(file_path, period_1, period_2)

		# Train/Test split
		train_size = int(features_array.shape[0] * split_ratio)
		test_size = features_array.shape[0] - train_size
		train_set = features_array[: train_size]
		test_set = features_array[train_size :]
		assert (train_set.shape[0] + test_set.shape[0]) == features_array.shape[0]

		# Normalize dataset
		norm_train_set, norm_test_set = normalize(train_set, test_set)
		assert norm_train_set.shape == train_set.shape
		assert norm_test_set.shape == test_set.shape

		# Save file
		np.save(DATA_PATH + stock_name + "_train_data" + "_p1" + str(period_1) + "_p2" + str(period_2) + ".npy", norm_train_set)
		np.save(DATA_PATH + stock_name + "_test_data" + "_p1" + str(period_1) + "_p2" + str(period_2) +".npy", norm_test_set)

def main():
	parser = argparse.ArgumentParser()
	# File path to specific file, if this is not provided, the script will process everything in the data directory with the right format
	parser.add_argument("--file_path", type = str, default = None)
	# Train test split ratio
	parser.add_argument("--split_ratio", type = float, default = 0.8)
	# Averaging period
	parser.add_argument("--period_1", type = int, default = 10)
	# For MACD
	parser.add_argument("--period_2", type = int, default = 25)
	args = parser.parse_args()

	create_dataset(args.file_path, args.split_ratio, args.period_1, args.period_2)

if __name__ == '__main__':
	main()
