import torch
import torch.nn.functional as F
from models.a3c_lstm import *

class Agent(object):
	def __init__(self, sdae_model, model, args):
		self.sdae_model = sdae_model
		self.model = model
		self.eps_len = 0
		self.args = args
		self.values = []
		self.log_probs = []
		self.rewards = []
		self.entropies = []
		self.info = None
		self.reward = 0
		self.gpu_id = -1

	def select_action(self, state, lstm_inputs, training = True):
		env_state, private_state = state
		env_state = torch.from_numpy(env_state).float()
		private_state = torch.from_numpy(private_state).float()
		hx, cx = lstm_inputs
		if self.gpu_id >= 0:
			with torch.cuda.device(self.gpu_id):
				env_state = env_state.cuda()
				private_state = private_state.cuda()
				hx = hx.cuda()
				cx = cs.cuda()
		if(training):
			with torch.no_grad():
				sdae_state = self.sdae_model(env_state, training = False)
			value, logit, (hx, cx) = self.model((torch.cat((sdae_state, private_state)).unsqueeze(0), (hx, cx)))
			prob = F.softmax(logit, dim=1)
			log_prob = F.log_softmax(logit, dim=1)
			entropy = -(log_prob * prob).sum(1)
			action = prob.multinomial(1).data
			log_prob = log_prob.gather(1, action)
			return action, value, log_prob, entropy, (hx, cx)
		else:
			# testing
			with torch.no_grad():
				sdae_state = self.sdae_model(env_state, training = False)
				value, logit, (hx, cx) = self.model((torch.cat((sdae_state, private_state)), (hx, cx)))
			prob = F.softmax(logit, dim=1)
			action = prob.max(1)[1].data.cpu().numpy()
			return action[0], (hx, cx)

	def step(self, value, log_prob, entropy, reward):
		self.values.append(value)
		self.log_probs.append(log_prob)
		self.rewards.append(reward)
		self.entropies.append(entropy)

	def clear_values(self):
		self.values = []
		self.log_probs = []
		self.rewards = []
		self.entropies = []
