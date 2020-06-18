import torch
import torch.nn.functional as F
from models.a3c_lstm import *

class Agent(object):
     def __init__(self, model, args):
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
        state = torch.from_numpy(state).float()
        hx, cx = lstm_inputs
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                state = state.cuda()
                hx = hx.cuda()
                cx = cs.cuda()
        if(training):
            value, logit, (hx, cx) = self.model((Variable(state.unsqueeze(0)), (hx, cx)))
            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)
            entropy = -(log_prob * prob).sum(1)
            action = prob.multinomial(1).data
            log_prob = log_prob.gather(1, Variable(action))
            return action, value, log_prob, entropy, (hx, cs)
        else:
            # testing
            with torch.no_grad():
                value, logit, (self.hx, self.cx) = self.model((Variable(state.unsqueeze(0)), (hx, cx)))
            prob = F.softmax(logit, dim=1)
            action = prob.max(1)[1].data.cpu().numpy()
            return action[0], (hx, cs)

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
