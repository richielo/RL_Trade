import os
import sys
import numpy as np

class Single_Stock_Env():
    def __init__(self, stock_raw_data, stock_norm_data, starting_capital, min_episode_length, max_episode_length, max_position, trans_cost_rate = 0.0005, slippage_rate = 0.001, full_data_episode = False):
        self.stock_raw_data = stock_raw_data
        self.stock_mean_vec = np.mean(stock_raw_data[:, : -3], axis = 0)
        self.stock_std_vec = np.std(stock_raw_data[:, : -3], axis = 0)
        self.stock_norm_data = stock_norm_data
        self.starting_capital = starting_capital
        self.min_episode_length = min_episode_length
        self.max_episode_length = max_episode_length
        self.max_position = max_position
        self.num_actions = int(self.max_position * 2 + 1)
        self.trans_cost_rate = trans_cost_rate
        self.slippage_rate = slippage_rate
        self.full_data_episode = full_data_episode
        self.init_episode()

    def init_episode(self):
        if(self.full_data_episode is False):
            self.curr_eps_length = np.random.randint(self.min_episode_length, self.max_episode_length + 1)
            self.eps_start_point = np.random.randint(0, self.stock_raw_data.shape[0] - self.curr_eps_length)
            self.curr_eps_raw_data = self.stock_raw_data[self.eps_start_point : self.eps_start_point + self.curr_eps_length]
            self.curr_eps_norm_data = self.stock_norm_data[self.eps_start_point : self.eps_start_point + self.curr_eps_length]
        else:
            self.curr_eps_length = self.stock_raw_data.shape[0]
            self.curr_eps_raw_data = self.stock_raw_data
            self.curr_eps_norm_data = self.stock_norm_data
        self.curr_state_index = 1
        self.curr_raw_state = self.curr_eps_raw_data[self.curr_state_index]
        self.curr_norm_state = self.curr_eps_norm_data[self.curr_state_index]
        self.curr_capital = self.starting_capital
        self.last_action = np.zeros(self.num_actions)
        # Volume, Average Bought Price - latter kind of useless
        self.curr_holdings = [0.0, 0.0]
        self.done = False

    def step(self, action):
        og_position = float(self.curr_holdings[0])
        position_change = float(action) - self.max_position
        position_changed = False
        if(position_change > 0):
            # Buy action
            cost = self.curr_raw_state[6] * position_change
            if(self.curr_capital >= cost):
                # Buying is performed
                new_average_price = (self.curr_holdings[0] * self.curr_holdings[1] + cost) / (self.curr_holdings[0] + position_change)
                self.curr_holdings = [self.curr_holdings[0] + position_change, new_average_price]
                self.curr_capital -= cost
                position_changed = True

        elif(position_change == 0):
            # Do nothing
            position_changed = False
        else:
            # Sell action
            num_curr_holdings = self.curr_holdings[0]
            if(num_curr_holdings >= abs(position_change)):
                # Selling is performed
                self.curr_holdings = [self.curr_holdings[0] + position_change, self.curr_holdings[1]]
                self.curr_capital += self.curr_raw_state[6] * abs(position_change)
                position_changed = True


        # Reward
        if(position_changed):
            reward = (self.curr_raw_state[6] - self.curr_eps_raw_data[self.curr_state_index - 1][6]) * og_position - (self.trans_cost_rate + self.slippage_rate) * abs(position_change)
            #reward = (next_raw_state[6] - self.curr_raw_state[6]) * self.curr_holdings[0] - (self.trans_cost_rate + self.slippage_rate) * abs(position_change)
        else:
            reward = (self.curr_raw_state[6] - self.curr_eps_raw_data[self.curr_state_index - 1][6]) * og_position
            #reward = (next_raw_state[6] - self.curr_raw_state[6]) * self.curr_holdings[0]

        # Next state
        self.curr_state_index += 1
        next_raw_state = self.curr_eps_raw_data[self.curr_state_index]
        next_norm_state = self.curr_eps_norm_data[self.curr_state_index]

        """
        # Reward -- old
        if(position_changed):
            reward = (next_raw_state[6] - self.curr_raw_state[6]) * og_position - (self.trans_cost_rate + self.slippage_rate) * abs(position_change)
            #reward = (next_raw_state[6] - self.curr_raw_state[6]) * self.curr_holdings[0] - (self.trans_cost_rate + self.slippage_rate) * abs(position_change)
        else:
            reward = (next_raw_state[6] - self.curr_raw_state[6]) * og_position
            #reward = (next_raw_state[6] - self.curr_raw_state[6]) * self.curr_holdings[0]
        """
        # Done flag
        if(self.curr_state_index == self.curr_eps_raw_data.shape[0] - 1):
            self.done = True

        # Update environment variables
        self.curr_raw_state = next_raw_state
        self.curr_norm_state = next_norm_state
        self.last_action = np.zeros(self.num_actions)
        self.last_action[action] = 1

        #TODO fix state representation, change sharpe ratio to 0.0 if no holdings, also add trading capital as feature

        # Returns reward, and next states
        return reward, self.get_current_input_to_model(), self.done

    def get_current_input_to_model(self):
        # Return current normalized state for the agent to use

        # Remove timestamp
        curr_input_state = self.curr_norm_state[:11]

        # Check holdings to update sharp ratio. If holdings is 0, set zero
        if(self.curr_holdings[0] == 0):
            curr_input_state[-1] = 0.0

        # Append trading capital and stock/cash ratio
        sc_ratio = (self.curr_holdings[0] * self.curr_raw_state[6]) / self.curr_capital

        #TODO: change it to tensor to avoid conversion
        #return np.concatenate([curr_input_state, np.array([self.curr_capital, sc_ratio])])
        return (curr_input_state, np.array([self.curr_capital, sc_ratio]))

    def calc_total_portfolio_value(self):
        return self.curr_holdings[0] * self.curr_raw_state[6] + self.curr_capital

    def reset(self):
        self.init_episode()
