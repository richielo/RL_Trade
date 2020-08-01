import numpy as np
import sys
import torch
import torch.optim as optim
from environments.single_stock_env import Single_Stock_Env
from models.utils import ensure_shared_grads
from models.a3c_lstm import A3C_LSTM
from a3c_lstm_agent import Agent
from torch.autograd import Variable

LSTM_SIZE = 128

def train(rank, args, sdae_model, shared_model, optimizer, env_config, train_process_finish_flags):
	# Environment variables
	stock_raw_data = env_config['stock_raw_data']
	stock_norm_data = env_config['stock_norm_data']
	starting_capital = env_config['starting_capital']
	min_episode_length = env_config['min_episode_length']
	max_episode_length = env_config['max_episode_length']
	max_position = env_config['max_position']
	trans_cost_rate = env_config['trans_cost_rate']
	slippage_rate = env_config['slippage_rate']

	gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
	# Set seed
	torch.manual_seed(args.seed + rank)
	if gpu_id >= 0:
		torch.cuda.manual_seed(args.seed + rank)
	np.random.seed(args.seed + rank)

	# Initialize environment
	if(trans_cost_rate is not None and slippage_rate is not None):
		env = Single_Stock_Env(stock_raw_data, stock_norm_data, starting_capital, min_episode_length, max_episode_length, max_position, trans_cost_rate, slippage_rate)
	else:
		env = Single_Stock_Env(stock_raw_data, stock_norm_data, starting_capital, min_episode_length, max_episode_length, max_position)
	state = env.get_current_input_to_model()

	# Initialize optimizers
	if optimizer is None:
		if args.optimizer_type == 'RMSprop':
			optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
		if args.optimizer_type == 'Adam':
			optimizer = optim.Adam(shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
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
	eps_num = 0
	total_steps = 0
	while True:
		if gpu_id >= 0:
			with torch.cuda.device(gpu_id):
				agent.model.load_state_dict(shared_model.state_dict())
		else:
			agent.model.load_state_dict(shared_model.state_dict())

		# For truncating LSTM's update
		if env.done:
			if gpu_id >= 0:
				with torch.cuda.device(gpu_id):
					hx = Variable(torch.zeros(1, LSTM_SIZE).cuda())
					cx = Variable(torch.zeros(1, LSTM_SIZE).cuda())
			else:
				hx = Variable(torch.zeros(1, LSTM_SIZE))
				cx = Variable(torch.zeros(1, LSTM_SIZE))
		else:
			hx = Variable(hx.data)
			cx = Variable(cx.data)

		for step in range(args.num_steps):
			action, val, log_prob, entropy, (next_hx, next_cx) = agent.select_action(state, (hx, cx))
			#print("Before act")
			#print(env.curr_holdings)
			#print(env.curr_capital)
			#print(state)
			#reward, next_state, _ = env.step(6 if step == 0 else 1)
			reward, next_state, _ = env.step(action)
			#print("After act")
			#print(env.curr_holdings)
			#print(env.curr_capital)
			#print(next_state)
			#print(reward)
			#if(step == 1):
			#	exit()
			agent.step(val, log_prob, entropy, reward)
			state = next_state
			(hx, cx) = (next_hx, next_cx)
			total_steps += 1

			if((total_steps % 500000) == 0):
				print("Rank: " + str(rank) + " | Training episode: " + str(eps_num) + " | Total steps: " + str(total_steps))
				sys.stdout.flush()
			if(total_steps >= args.num_train_steps):
				break
			if env.done:
				break

		R = torch.zeros(1, 1).float()
		# Get values of current state if the episode is not done
		if not env.done:
			env_state, private_state = state
			env_state = torch.from_numpy(env_state).float()
			private_state = torch.from_numpy(private_state).float()
			if gpu_id >= 0:
				with torch.cuda.device(gpu_id):
					env_state = env_state.cuda()
					private_state = private_state.cuda()
			with torch.no_grad():
				sdae_state = agent.sdae_model(env_state, training = False)
			value, _, _ = agent.model((Variable(torch.cat((sdae_state, private_state)).unsqueeze(0)), (hx, cx)))
			R = value.data

		if gpu_id >= 0:
			with torch.cuda.device(gpu_id):
				R = R.cuda()

		agent.values.append(Variable(R))
		policy_loss = 0
		value_loss = 0
		gae = torch.zeros(1, 1)
		if gpu_id >= 0:
			with torch.cuda.device(gpu_id):
				gae = gae.cuda()
		R = Variable(R)
		for i in reversed(range(len(agent.rewards))):
			R = args.gamma * R + agent.rewards[i]
			advantage = R - agent.values[i]
			value_loss = value_loss + 0.5 * advantage.pow(2)

			# Generalized Advantage Estimataion
			delta_t = agent.rewards[i] + args.gamma * agent.values[i + 1].data - agent.values[i].data

			gae = gae * args.gamma * args.tau + delta_t

			policy_loss = policy_loss - agent.log_probs[i] * Variable(gae) - 0.01 * agent.entropies[i]

		agent.model.zero_grad()
		(policy_loss + 0.5 * value_loss).backward()
		ensure_shared_grads(agent.model, shared_model, gpu = gpu_id >= 0)
		optimizer.step()
		agent.clear_values()

		if env.done:
			eps_num += 1
			env.reset()
			state = env.get_current_input_to_model()

		if(total_steps >= args.num_train_steps):
			train_process_finish_flags[rank] = 1
			print("Train worker " + str(rank) + " done")
			sys.stdout.flush()
			break
