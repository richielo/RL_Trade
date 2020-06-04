import os
import sys
import json
import pandas as pd
from feature_extraction_helpers import *

DATA_PATH = "data/"
TEST_FILE_NAME = "final_AAL_api_call_past_20_years.txt_output.json"

def extract_timestamp(item):
	try:
		return int(item['t'])
	except KeyError:
		return 0

with open(DATA_PATH + TEST_FILE_NAME) as test_data_file:
	data = json.load(test_data_file)
	"""
	print(data['ticker'])
	print(data['queryCount'])
	print(data['resultsCount'])
	print(data['adjusted'])
	print(len(data['results']))
	print(data['results'][0])
	print(data['results'][1])
	"""
	result_json_list = data['results']
	before = len(result_json_list)
	print(len(result_json_list))
	result_dataframe = pd.DataFrame.from_records(result_json_list)
	all_zeros_rows = result_dataframe[(result_dataframe['o'] ==0) | (result_dataframe['c'] == 0) | (result_dataframe['h'] == 0) | ((result_dataframe['l'] == 0))].index
	print(len(all_zeros_rows))
	result_dataframe.drop(all_zeros_rows, inplace = True)
	result_json_list = json.loads(result_dataframe.to_json(orient = 'records'))
	after = len(result_json_list)
	print("Shit ratio: " + str((before - after) / float(before) * 100))
	# Sort by timestamp
	result_json_list.sort(key = extract_timestamp)
	#exit()
	#print(result_json_list[0])
	#print(result_json_list[1])
	#print(result_json_list[2])
	#print(result_json_list[3])
	#exit()
	# Check timestamp difference between adjacent data points
	"""
	diff_dict = {}
	for i in range(1, len(result_json_list)):
		diff = data['results'][i]['t'] - data['results'][i-1]['t']
		if(diff not in diff_dict.keys()):
			diff_dict[diff] = 1
		else:
			diff_dict[diff] += 1
	print(diff_dict)
	print(data['resultsCount'])
	"""
	# Check feature extraction
	ROC(result_json_list, 10)
	ATR(result_json_list, 10)
	MA(result_json_list, 10)
	EMA(result_json_list, 12)
	MACD(result_json_list, 12, 26)
