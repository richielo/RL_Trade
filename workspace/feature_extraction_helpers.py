import os
import sys
import numpy as np

"""
TODO: All these feature extraction functions take the closing price at the moment. Once all the codes are ready, we should tinker with this and see which is better
"""
# Moving Average
def MA(sorted_data, period):
	ma_list = []
	for i in range(len(sorted_data)):
		if(i < period - 1):
			ma_list.append(sorted_data[i]['c'])
			#ma_list.append(None)
		else:
			ma_sum = 0.0
			for j in range(i, i - period, -1):
				ma_sum += sorted_data[j]['c']
			curr_ma = ma_sum / period
			ma_list.append(curr_ma)
	return ma_list

# Exponential Moving Average
def EMA(sorted_data, period):
	sma = np.mean([ item['c'] for item in sorted_data[: period]])
	ema_list = []
	smooth_weighting = 2.0 / (period + 1)
	for i in range(len(sorted_data)):
		if(i < period - 1):
			ema_list.append(sma)
			#ema_list.append(None)
		else:
			new_ema = (sorted_data[i]['c'] - ema_list[-1]) * smooth_weighting + ema_list[-1]
			ema_list.append(new_ema)
	assert len(ema_list) == len(sorted_data)
	return np.array(ema_list)

# Moving Average Convergence Divergence
def MACD(sorted_data, period_1, period_2):
	period_1_ema_list = EMA(sorted_data, period_1)
	period_2_ema_list = EMA(sorted_data, period_2)
	assert len(period_1_ema_list) == len(period_2_ema_list)
	macd_list = period_1_ema_list - period_2_ema_list
	return macd_list

# True range
def TR(sorted_data, curr_index):
	ch_pc = abs(sorted_data[curr_index]['h'] - sorted_data[curr_index - 1]['c'])
	cl_pc = abs(sorted_data[curr_index]['l'] - sorted_data[curr_index - 1]['c'])
	ch_cl = abs(sorted_data[curr_index]['h'] - sorted_data[curr_index]['l'])
	return max([ch_pc, cl_pc, ch_cl])

# Average True Range
def ATR(sorted_data, period):
	# Get true ranges first
	tr_list = []
	for i in range(len(sorted_data)):
		if(i == 0):
			# Can only use current high and current low
			tr_list.append(abs(sorted_data[i]['h'] - sorted_data[i]['l']))
		else:
			tr_list.append(TR(sorted_data, i))
	# Get average true_range
	atr_list = []
	for i in range(len(tr_list)):
		if(i < period - 1):
			atr_list.append(sum(tr_list[: i + 1]) / len(tr_list[: i + 1]))
			#atr_list.append(None)
		else:
			atr_list.append(sum(tr_list[i- period + 1: i + 1]) / period)
	assert(len(tr_list) == len(atr_list))
	return np.array(atr_list)

"""
This function would fail if there are zeros, remove those first
"""
# Price Rate of Change
def ROC(sorted_data, period):
	roc_list = []
	for i in range(len(sorted_data)):
		if(i == 0):
			roc_list.append(0.0)
			#roc_list.append(None)
		elif(i < period):
			curr_roc = (sorted_data[i]['c'] - sorted_data[0]['c']) / sorted_data[0]['c']
			roc_list.append(curr_roc)
			#roc_list.append(None)
		else:
			curr_roc = (sorted_data[i]['c'] - sorted_data[i - period]['c']) / sorted_data[i - period]['c']
			roc_list.append(curr_roc)
	return roc_list

# Sharpe Ratio
def SR(sorted_data, period, risk_free_rate = 0.007):
	# 6.5 trading hours per day
	risk_free_rate = risk_free_rate / (6.5 * 60)
	sr_list = []
	for i in range(len(sorted_data)):
		if (i < period):
			sr_list.append(0.0)
		else:
			expected_return_list = []
			for j in range(i, i - period, -1):
				 perc_change = (sorted_data[j]['c'] - sorted_data[j-1]['c']) / sorted_data[j-1]['c']
				 expected_return_list.append(perc_change)
			expected_return_list = np.array(expected_return_list)
			std = np.std(expected_return_list)
			if(std != 0.0):
				sr_list.append((np.mean(expected_return_list) - risk_free_rate) / np.std(expected_return_list))
			else:
				sr_list.append(0.0)
	return sr_list

# Extract market variables into lists
def Market_Variables(sorted_data):
	o_list = []
	c_list = []
	h_list = []
	l_list = []
	v_list = []
	t_list = []
	for i in range(len(sorted_data)):
		o_list.append(sorted_data[i]['o'])
		c_list.append(sorted_data[i]['c'])
		h_list.append(sorted_data[i]['h'])
		l_list.append(sorted_data[i]['l'])
		v_list.append(sorted_data[i]['v'])
		t_list.append(sorted_data[i]['t'])
	return o_list, c_list, h_list, l_list, v_list, t_list
