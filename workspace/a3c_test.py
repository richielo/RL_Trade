import numpy as np
import sys
from collections import Counter
import torch
import torch.optim as optim
from environments.single_stock_env import Single_Stock_BS_Env, Single_Stock_Full_Env
from models.utils import ensure_shared_grads
from models.a3c_lstm import A3C_LSTM
from a3c_lstm_agent import Agent
from torch.autograd import Variable
import matplotlib.pyplot as plt


LSTM_SIZE = 128
RESULT_DATA_PATH = "trained_data/"

def test(args, sdae_model, shared_model, env_config, train_process_finish_flags):
	# Environment variables
	stock_raw_data = env_config['stock_raw_data']
	stock_norm_data = env_config['stock_norm_data']
	starting_capital = env_config['starting_capital']
	min_episode_length = env_config['min_episode_length']
	max_episode_length = env_config['max_episode_length']
	max_position = env_config['max_position']
	trans_cost_rate = env_config['trans_cost_rate']
	slippage_rate = env_config['slippage_rate']

	gpu_id = args.gpu_ids[-1]
	# Set seed
	torch.manual_seed(args.seed)
	if gpu_id >= 0:
		torch.cuda.manual_seed(args.seed)
	np.random.seed(args.seed)

	# Initialize environment
	if(trans_cost_rate is not None and slippage_rate is not None):
		if(args.full_env):
			env = Single_Stock_Full_Env(stock_raw_data, stock_norm_data, starting_capital, min_episode_length, max_episode_length, max_position, trans_cost_rate, slippage_rate, full_data_episode = True)
		else:
			env = Single_Stock_BS_Env(stock_raw_data, stock_norm_data, starting_capital, min_episode_length, max_episode_length, max_position, trans_cost_rate, slippage_rate, full_data_episode = True)
	else:
		if(args.full_env):
			env = Single_Stock_Full_Env(stock_raw_data, stock_norm_data, starting_capital, min_episode_length, max_episode_length, max_position, full_data_episode = True)
		else:
			env = Single_Stock_BS_Env(stock_raw_data, stock_norm_data, starting_capital, min_episode_length, max_episode_length, max_position, full_data_episode = True)
	state = env.get_current_input_to_model()
	agent_model = A3C_LSTM(args.rl_input_dim, args.num_actions)
	agent = Agent(sdae_model, agent_model, args)
	agent.gpu_id = gpu_id

	cx = Variable(torch.zeros(1, LSTM_SIZE))
	hx = Variable(torch.zeros(1, LSTM_SIZE))

	if gpu_id >= 0:
		with torch.cuda.device(gpu_id):
			agent.model = agent.model.cuda()
			agent.model.train()
			cx = cx.cuda()
			hx = hx.cuda()
			state = state.cuda()
	test_num = 0
	reward_list = []
	final_equity_list = []
	max_reward = -1e10
	# If all training processes have ended this will be True. Then, one more run would be done to capture final result
	terminate_next_iter = False
	while True:
		if gpu_id >= 0:
			with torch.cuda.device(gpu_id):
				agent.model.load_state_dict(shared_model.state_dict())
		else:
			agent.model.load_state_dict(shared_model.state_dict())

		episodic_reward = 0.0
		count = 0
		actions = []
		rewards = []
		pv_list = []
		pv_change_list = []
		while env.done is False:
			action, (next_hx, next_cx) = agent.select_action(state, (hx, cx), training = False)
			actions.append(action - 3)
			reward, next_state, _ = env.step(action)
			"""
			rewards.append(reward)
			pv_list.append(env.calc_total_portfolio_value())
			if(count == 0):
				pv_change_list.append(0.0)
			else:
				pv_change_list.append(pv_list[count] - pv_list[count - 1])
			"""
			episodic_reward += reward
			state = next_state
			(hx, cx) = (next_hx, next_cx)
			count += 1
		index_list = [i for i in range(1, len(pv_list) + 1)]
		"""
		#print(pv_list)
		print(max(pv_list))
		print(min(pv_list))
		print(sum(rewards))
		fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
		ax1.plot(index_list, pv_list)
		ax2.plot(index_list, rewards)
		ax3.plot(index_list, pv_change_list)
		plt.show()
		exit()
		"""
		# Results logging
		reward_list.append(episodic_reward)
		port_value = env.calc_total_portfolio_value()
		final_equity_list.append(port_value)

		test_num += 1
		#print("Test num: " + str(test_num) + " | Test reward: " + str(episodic_reward) + " | Final equity: " + str(port_value))
		#print(env.curr_holdings)
		print("Test num: {0} | Test reward: {1} | Holdings: {2} | End Capital: {3} | Final equity : {4}".format(test_num, episodic_reward, env.curr_holdings[0], env.curr_capital, port_value))
		print(Counter(actions))
		print("\n")
		sys.stdout.flush()
		env.reset()
		state = env.get_current_input_to_model()
		if gpu_id >= 0:
			with torch.cuda.device(gpu_id):
				hx = Variable(torch.zeros(1, LSTM_SIZE).cuda())
				cx = Variable(torch.zeros(1, LSTM_SIZE).cuda())
				state = state.cuda()
		else:
			hx = Variable(torch.zeros(1, LSTM_SIZE))
			cx = Variable(torch.zeros(1, LSTM_SIZE))

		# Save model
		if(args.use_filter_data):
			model_name = args.stock_env + "_p1" + str(args.period_1) + "_p2" + str(args.period_2) + "_minEL" + str(args.min_episode_length) + "_maxEL" + str(args.max_episode_length) + "_nstep" + str(args.num_steps) + "_ntrainstep" + str(args.num_train_steps) + "_lr" + str(args.lr) + "_gamma" + str(args.gamma) + "_tau" + str(args.tau) + "_best_filtered_fyear" + str(args.filter_by_year) + "_full" if args.full_env else "" +  ".pt"
		else:
			model_name = args.stock_env + "_p1" + str(args.period_1) + "_p2" + str(args.period_2) + "_minEL" + str(args.min_episode_length) + "_maxEL" + str(args.max_episode_length) + "_nstep" + str(args.num_steps) + "_ntrainstep" + str(args.num_train_steps) + "_lr" + str(args.lr) + "_gamma" + str(args.gamma) + "_tau" + str(args.tau) + "_full" if args.full_env else "" + "_best.pt"
		if(terminate_next_iter):
			if(args.use_filter_data):
				model_name = args.stock_env + "_p1" + str(args.period_1) + "_p2" + str(args.period_2) + "_minEL" + str(args.min_episode_length) + "_maxEL" + str(args.max_episode_length) + "_nstep" + str(args.num_steps) + "_ntrainstep" + str(args.num_train_steps) + "_lr" + str(args.lr) + "_gamma" + str(args.gamma) + "_tau" + str(args.tau) + "_final_filtered_fyear" + str(args.filter_by_year) + "_full" if args.full_env else "" +  ".pt"
			else:
				model_name = args.stock_env + "_p1" + str(args.period_1) + "_p2" + str(args.period_2) + "_minEL" + str(args.min_episode_length) + "_maxEL" + str(args.max_episode_length) + "_nstep" + str(args.num_steps) + "_ntrainstep" + str(args.num_train_steps) + "_lr" + str(args.lr) + "_gamma" + str(args.gamma) + "_tau" + str(args.tau) + "_full" if args.full_env else "" + "_final.pt"
			if gpu_id >= 0:
				with torch.cuda.device(gpu_id):
					state_to_save = agent.model.state_dict()
					torch.save(state_to_save, '{0}{1}.dat'.format(args.save_model_dir, model_name))
			else:
				state_to_save = agent.model.state_dict()
				torch.save(state_to_save, '{0}{1}.dat'.format(args.save_model_dir, model_name))
			print("saved final")
			break
		else:
			if(episodic_reward > max_reward):
				#model_name = args.stock_env + "_p1" + str(args.period_1) + "_p2" + str(args.period_2) + "_minEL" + str(args.min_episode_length) + "_maxEL" + str(args.max_episode_length) + "_nstep" + str(args.num_steps) + "_ntrainstep" + str(args.num_train_steps) + "_lr" + str(args.lr) + "_gamma" + str(args.gamma) + "_tau" + str(args.tau) + "_best.pt"
				if gpu_id >= 0:
					with torch.cuda.device(gpu_id):
						state_to_save = agent.model.state_dict()
						torch.save(state_to_save, '{0}{1}.dat'.format(args.save_model_dir, model_name))
				else:
					state_to_save = agent.model.state_dict()
					torch.save(state_to_save, '{0}{1}.dat'.format(args.save_model_dir, model_name))

		# Save results
		if(args.use_filter_data):
			np.save(RESULT_DATA_PATH + "epi_reward_filtered_" + model_name, np.array(reward_list))
			np.save(RESULT_DATA_PATH + "portfolio_filtered_" + model_name, np.array(final_equity_list))
		else:
			np.save(RESULT_DATA_PATH + "epi_reward_" + model_name, np.array(reward_list))
			np.save(RESULT_DATA_PATH + "portfolio_" + model_name, np.array(final_equity_list))

		if(torch.all(train_process_finish_flags == torch.ones(train_process_finish_flags.size(0)))):
			terminate_next_iter = True
			print("From test process: all training process terminated")
			sys.stdout.flush()

def test_one_episode(args, sdae_model, shared_model, env_config):
	# Environment variables
	stock_raw_data = env_config['stock_raw_data']
	stock_norm_data = env_config['stock_norm_data']
	starting_capital = env_config['starting_capital']
	min_episode_length = env_config['min_episode_length']
	max_episode_length = env_config['max_episode_length']
	max_position = env_config['max_position']
	trans_cost_rate = env_config['trans_cost_rate']
	slippage_rate = env_config['slippage_rate']

	gpu_id = args.gpu_ids[-1]
	# Set seed
	torch.manual_seed(args.seed)
	if gpu_id >= 0:
		torch.cuda.manual_seed(args.seed)
	np.random.seed(args.seed)

	# Initialize environment
	if(trans_cost_rate is not None and slippage_rate is not None):
		if(args.full_env):
			env = Single_Stock_Full_Env(stock_raw_data, stock_norm_data, starting_capital, min_episode_length, max_episode_length, max_position, trans_cost_rate, slippage_rate, full_data_episode = True)
		else:
			env = Single_Stock_BS_Env(stock_raw_data, stock_norm_data, starting_capital, min_episode_length, max_episode_length, max_position, trans_cost_rate, slippage_rate, full_data_episode = True)
	else:
		if(args.full_env):
			env = Single_Stock_Full_Env(stock_raw_data, stock_norm_data, starting_capital, min_episode_length, max_episode_length, max_position, full_data_episode = True)
		else:
			env = Single_Stock_BS_Env(stock_raw_data, stock_norm_data, starting_capital, min_episode_length, max_episode_length, max_position, full_data_episode = True)
	state = env.get_current_input_to_model()
	agent_model = A3C_LSTM(args.rl_input_dim, args.num_actions)
	agent = Agent(sdae_model, agent_model, args)
	agent.gpu_id = gpu_id

	cx = Variable(torch.zeros(1, LSTM_SIZE))
	hx = Variable(torch.zeros(1, LSTM_SIZE))

	if gpu_id >= 0:
		with torch.cuda.device(gpu_id):
			agent.model = agent.model.cuda()
			agent.model.train()
			cx = cx.cuda()
			hx = hx.cuda()
			state = state.cuda()

	if gpu_id >= 0:
		with torch.cuda.device(gpu_id):
			agent.model.load_state_dict(shared_model.state_dict())
	else:
		agent.model.load_state_dict(shared_model.state_dict())

	episodic_reward = 0.0
	count = 0
	actions = []
	rewards = []
	pv_list = []
	pv_change_list = []
	while env.done is False:
		action, (next_hx, next_cx) = agent.select_action(state, (hx, cx), training = False)
		actions.append(action)
		reward, next_state, _ = env.step(action)
		rewards.append(reward)
		"""
		pv_list.append(env.calc_total_portfolio_value())
		if(count == 0):
			pv_change_list.append(0.0)
		else:
			pv_change_list.append(pv_list[count] - pv_list[count - 1])
		"""
		episodic_reward += reward
		state = next_state
		(hx, cx) = (next_hx, next_cx)
		count += 1
	index_list = [i for i in range(1, len(pv_list) + 1)]
	"""
	#print(pv_list)
	print(max(pv_list))
	print(min(pv_list))
	print(sum(rewards))
	fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
	ax1.plot(index_list, pv_list)
	ax2.plot(index_list, rewards)
	ax3.plot(index_list, pv_change_list)
	plt.show()
	exit()
	"""
	# Results logging
	port_value = env.calc_total_portfolio_value()
	#print("Test num: " + str(test_num) + " | Test reward: " + str(episodic_reward) + " | Final equity: " + str(port_value))
	#print(env.curr_holdings)
	print("Test reward: {0} | Holdings: {1}/{2} | End Capital: {3} | Final equity : {4}".format(episodic_reward, env.curr_holdings[0], env.curr_holdings[1], env.curr_capital, port_value))
	print("\n")
	sys.stdout.flush()
	return episodic_reward, rewards, actions
