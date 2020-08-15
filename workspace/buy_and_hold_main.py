import os
import sys
import time
import numpy as np
import argparse
from utils.data_utils import load_dataset
import matplotlib.pyplot as plt

"""

This script goes through the training and testing data to apply the buy and hold baseline
It buys at the first time step using all of the capital and holds it til the end. We record
the change in portfolio value. We don't need the stock environment.

"""

DATA_PATH = "data/"
RESULT_PATH = "results/BandH/"

parser = argparse.ArgumentParser(description='Buy&Hold')
parser.add_argument(
	'--stock_env',
	default='AAPL',
	metavar='ENV',
	help='stock environment to train on (default: AAPL)')
parser.add_argument(
	'--period_1',
	type=int,
	default=30,
	metavar='p1',
	help='main averaging period length of features')
parser.add_argument(
	'--period_2',
	type=int,
	default=60,
	metavar='p2',
	help='secondary averaging period length of features for MACD')
parser.add_argument(
	'--use_filter_data',
	default=True,
	metavar='AM',
	help='Whether to use filtered data')
parser.add_argument(
	'--filter_by_year',
	type=int,
	default=2000,
	help='The oldest year to include')

if __name__ == '__main__':
	args = parser.parse_args()

	# Create environment dictionary
	starting_capital = 100000
	starting_index = 1
	trans_cost_rate = 0.0005
	slippage_rate = 0.001
	(train_head_name, test_head_name), train_norm_data, train_raw_data, test_norm_data, test_raw_data = load_dataset(args.stock_env, args.period_1, args.period_2, args.use_filter_data, args.filter_by_year)

	train_portfolio_value = []
	test_portfolio_value = []

	# Training data
	train_start_index = starting_index
	train_holding = 0
	train_capital = starting_capital
	while(train_start_index < train_raw_data.shape[0]):
		if(train_start_index == 1):
			# Buy
			num_bought = starting_capital / (train_raw_data[train_start_index][6] * (1.0 + slippage_rate + trans_cost_rate))
			train_capital -= num_bought * train_raw_data[train_start_index][6] * (1.0 + slippage_rate + trans_cost_rate)
			train_holding = num_bought
		#print(train_capital + train_raw_data[train_start_index][6] * train_holding)
		train_portfolio_value.append(train_capital + train_raw_data[train_start_index][6] * train_holding)
		train_start_index += 1
	print(len(train_portfolio_value))
	print(train_portfolio_value[-1])
	print(train_raw_data.shape)
	# Testing data
	test_start_index = starting_index
	test_holding = 0
	test_capital = starting_capital
	test_prices = []
	test_indices = []
	while(test_start_index < test_raw_data.shape[0]):
		if(test_start_index == 1):
			# Buy
			num_bought = starting_capital / (test_raw_data[test_start_index][6] * (1.0 + slippage_rate + trans_cost_rate))
			test_capital -= num_bought * (test_raw_data[test_start_index][6] * (1.0 + slippage_rate + trans_cost_rate))
			test_holding = num_bought
		test_prices.append(test_raw_data[test_start_index][6])
		test_portfolio_value.append(test_capital + test_raw_data[test_start_index][6] * test_holding)
		test_start_index += 1
	print(len(test_portfolio_value))
	print(test_portfolio_value[-1])
	print(test_holding)
	print(test_raw_data.shape)

	plt.plot([i for i in range(1, len(test_prices) + 1)], test_prices)
	plt.show()
