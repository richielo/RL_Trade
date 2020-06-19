import numpy as np
import torch
import torch.optim as optim
from environments.single_stock_env import Single_Stock_Env
from models.utils import ensure_shared_grads
from models.a3c_lstm import A3C_LSTM
from a3c_lstm_agent import Agent
from torch.autograd import Variable


LSTM_SIZE = 128
RESULT_DATA_PATH = "trained_data/"

def test(args, sdae_model, shared_model, env_config):
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
		env = Single_Stock_Env(stock_raw_data, stock_norm_data, starting_capital, min_episode_length, max_episode_length, max_position, trans_cost_rate, slippage_rate, full_data_episode = True)
	else:
		env = Single_Stock_Env(stock_raw_data, stock_norm_data, starting_capital, min_episode_length, max_episode_length, max_position, full_data_episode = True)
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
	while True:
		if gpu_id >= 0:
			with torch.cuda.device(gpu_id):
				agent.model.load_state_dict(shared_model.state_dict())
		else:
			agent.model.load_state_dict(shared_model.state_dict())

		episodic_reward = 0.0
		count = 0
		while env.done is False:
			action, (next_hx, next_cx) = agent.select_action(state, (hx, cx), training = False)
			reward, next_state, _ = env.step(action)
			episodic_reward += reward
			state = next_state
			(hx, cx) = (next_hx, next_cx)
			count += 1
		# Results logging
		reward_list.append(episodic_reward)
		port_value = env.calc_total_portfolio_value()
		final_equity_list.append(port_value)
		print("Test reward: " + str(episodic_reward))
		print("Final equity: " + str(port_value))

		test_num += 1
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
		model_name = args.stock_env + "_p1" + str(args.period_1) + "_p2" + str(args.period_2) + "_minEL" + str(args.min_episode_length) + "_maxEL" + str(args.max_episode_length) + "_nstep" + str(args.num_steps) + "_lr" + str(args.lr) + "_gamma" + str(args.gamma) + "_tau" + str(args.tau) + ".pt"
		if(episodic_reward > max_reward):
			max_reward = episodic_reward
			model_name = args.stock_env + "_p1" + str(args.period_1) + "_p2" + str(args.period_2) + "_minEL" + str(args.min_episode_length) + "_maxEL" + str(args.max_episode_length) + "_nstep" + str(args.num_steps) + "_lr" + str(args.lr) + "_gamma" + str(args.gamma) + "_tau" + str(args.tau) + ".pt"
			if gpu_id >= 0:
				with torch.cuda.device(gpu_id):
					state_to_save = agent.model.state_dict()
					torch.save(state_to_save, '{0}{1}.dat'.format(args.save_model_dir, model_name))
			else:
				state_to_save = agent.model.state_dict()
				torch.save(state_to_save, '{0}{1}.dat'.format(args.save_model_dir, model_name))

		# Save results
		np.save(RESULT_DATA_PATH + "epi_reward_" + model_name, np.array(reward_list))
		np.save(RESULT_DATA_PATH + "portfolio_" + model_name, np.array(final_equity_list))