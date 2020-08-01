"""
This script looks for json data in the data directory to create training and testing set based on the 12 features
"""
import os
import sys
from datetime import datetime, timezone
import argparse
import numpy as np
import json
import pandas as pd
from feature_extraction_helpers import *
from utils.timezone_utils import *

DATA_PATH = "data/"
FILE_EXT = ".json"

def filter_data_by_time(sorted_data):
	removal_indices = []
	for i in range(len(sorted_data)):
		ts = int(sorted_data[i]['t']) / 1000
		utc_datetime = datetime.utcfromtimestamp(ts)
		nyc_datetime = utc_datetime.replace(tzinfo=timezone.utc).astimezone(Eastern)
		# 0930-1130 and 1300-1500
		if(nyc_datetime.hour >= 9 and nyc_datetime.hour < 12):
			if(nyc_datetime.hour == 9 and nyc_datetime.minute <= 30):
				removal_indices.append(i)
			elif(nyc_datetime.hour == 11 and nyc_datetime.minute > 30):
				removal_indices.append(i)
		elif(nyc_datetime.hour >= 13 and nyc_datetime.hour <= 15):
			if(nyc_datetime.minute > 0):
				removal_indices.append(i)
		else:
			removal_indices.append(i)
	# Remove
	for index in sorted(removal_indices, reverse=True):
		del sorted_data[index]
	return sorted_data


def extract_timestamp(item):
	try:
		return int(item['t'])
	except KeyError:
		return 0

def extract_features(file_path, period_1, period_2, filter_by_time):
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

		# filter out data based on time
		if(filter_by_time):
			result_json_list = filter_data_by_time(result_json_list)

		# Compute technical indicators
		roc_list = ROC(result_json_list, period_1)
		atr_list = ATR(result_json_list, period_1)
		ma_list = MA(result_json_list, period_1)
		ema_list = EMA(result_json_list, period_1)
		macd_list = MACD(result_json_list, period_1, period_2)
		sr_list = SR(result_json_list, period_1)

		# Extract market variables
		o_list, c_list, h_list, l_list, v_list, t_list= Market_Variables(result_json_list)

		# Data size check
		assert len(roc_list) == len(atr_list) == len(ma_list) == len(ema_list) == len(macd_list) == len(o_list) == len(c_list) == len(h_list) == len(l_list) == len(v_list) == len(t_list)

		features_array = np.array([roc_list, atr_list, ma_list, ema_list, macd_list, o_list, c_list, h_list, l_list, v_list, sr_list, t_list]).T
		return features_array


def normalize(train_set, test_set):
	# Ignore the last two column which are timestamp and sharpe ratio
	norm_train_set = train_set.copy()
	norm_test_set = test_set.copy()
	mean_vec = np.mean(train_set[:, : -3], axis = 0)
	mean_mat = np.repeat(np.expand_dims(mean_vec, axis = 0), norm_train_set.shape[0], axis = 0)
	sd_vec = np.std(train_set[:, : -3], axis = 0)
	sd_mat = np.repeat(np.expand_dims(sd_vec, axis = 0), norm_train_set.shape[0], axis = 0)
	norm_train_set[:, : -3] = (train_set[:, : -3] - mean_mat) / sd_mat
	norm_test_set[:, : -3] = (test_set[:, : -3] - mean_mat[:norm_test_set.shape[0]]) / sd_mat[:norm_test_set.shape[0]]
	return norm_train_set, norm_test_set

def create_dataset(file_path, split_ratio, period_1, period_2, filter_by_time):
	if(file_path is None):
		# Go through the entire directory
		for file in os.listdir(DATA_PATH):
			if(file.endswith(FILE_EXT)):
				stock_name = file.split('_')[1]
				file_path = os.path.join(DATA_PATH, file)
				# Extract features
				features_array = extract_features(file_path, period_1, period_2, filter_by_time)

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
				if(filter_by_time):
					np.save(DATA_PATH + stock_name + "_train_data" + "_p1" + str(period_1) + "_p2" + str(period_2) + "_normalized_filtered.npy", norm_train_set)
					np.save(DATA_PATH + stock_name + "_train_data" + "_p1" + str(period_1) + "_p2" + str(period_2) + "_raw_filtered.npy", train_set)
					np.save(DATA_PATH + stock_name + "_test_data" + "_p1" + str(period_1) + "_p2" + str(period_2) +"_normalized_filtered.npy", norm_test_set)
					np.save(DATA_PATH + stock_name + "_test_data" + "_p1" + str(period_1) + "_p2" + str(period_2) +"_raw_filtered.npy", test_set)
				else:
					np.save(DATA_PATH + stock_name + "_train_data" + "_p1" + str(period_1) + "_p2" + str(period_2) + "_normalized.npy", norm_train_set)
					np.save(DATA_PATH + stock_name + "_train_data" + "_p1" + str(period_1) + "_p2" + str(period_2) + "_raw.npy", train_set)
					np.save(DATA_PATH + stock_name + "_test_data" + "_p1" + str(period_1) + "_p2" + str(period_2) +"_normalized.npy", norm_test_set)
					np.save(DATA_PATH + stock_name + "_test_data" + "_p1" + str(period_1) + "_p2" + str(period_2) +"_raw.npy", test_set)
	else:
		# Process individual file
		# Assume the file in the DATA_PATH directory
		stock_name = file_path.split('/')[1].split('_')[1]
		# Extract features
		features_array = extract_features(file_path, period_1, period_2, filter_by_time)

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
		print("train set size: " + str(norm_train_set.shape[0]))
		print("test set size: " + str(norm_test_set.shape[0]))

		# Save file
		if(filter_by_time):
			np.save(DATA_PATH + stock_name + "_train_data" + "_p1" + str(period_1) + "_p2" + str(period_2) + "_normalized_filtered.npy", norm_train_set)
			np.save(DATA_PATH + stock_name + "_train_data" + "_p1" + str(period_1) + "_p2" + str(period_2) + "_raw_filtered.npy", train_set)
			np.save(DATA_PATH + stock_name + "_test_data" + "_p1" + str(period_1) + "_p2" + str(period_2) +"_normalized_filtered.npy", norm_test_set)
			np.save(DATA_PATH + stock_name + "_test_data" + "_p1" + str(period_1) + "_p2" + str(period_2) +"_raw_filtered.npy", test_set)
		else:
			np.save(DATA_PATH + stock_name + "_train_data" + "_p1" + str(period_1) + "_p2" + str(period_2) + "_normalized.npy", norm_train_set)
			np.save(DATA_PATH + stock_name + "_train_data" + "_p1" + str(period_1) + "_p2" + str(period_2) + "_raw.npy", train_set)
			np.save(DATA_PATH + stock_name + "_test_data" + "_p1" + str(period_1) + "_p2" + str(period_2) +"_normalized.npy", norm_test_set)
			np.save(DATA_PATH + stock_name + "_test_data" + "_p1" + str(period_1) + "_p2" + str(period_2) +"_raw.npy", test_set)

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
	# Whether to filter data by time according to paper - only keep 0930-1130 and 1300-1500
	parser.add_argument('--filter_by_time', default=False, metavar='L', help='whether to filter data by time')
	args = parser.parse_args()

	create_dataset(args.file_path, args.split_ratio, args.period_1, args.period_2, args.filter_by_time)

if __name__ == '__main__':
	main()
