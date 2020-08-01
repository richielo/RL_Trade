import os
import time
import numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from models.a3c_lstm import A3C_LSTM
from models.sdae import SDAE
from models.shared_optimizers import SharedRMSprop, SharedAdam
from a3c_train import train
from a3c_test import test_one_episode
import matplotlib.pyplot as plt

DATA_PATH = "data/"
SDAE_PATH = "sdae_models/"

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0000001,
    metavar='LR',
    help='learning rate (default: 0.00001)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.999,
    metavar='G',
    help='discount factor for rewards (default: 0.999)')
parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--workers',
    type=int,
    default=1,
    metavar='W',
    help='how many training processes to use (default: 32)')
parser.add_argument(
    '--num_steps',
    type=int,
    default=240,
    metavar='NS',
    help='number of forward steps in A3C (default: 20)')
parser.add_argument(
    '--max_episode_length',
    type=int,
    default=500000,
    metavar='MaxEL',
    help='maximum length of an episode (default: 10000)')
parser.add_argument(
    '--min_episode_length',
    type=int,
    default=100000,
    metavar='MinPL',
    help='minimum length of an episode (default: 1000)')
parser.add_argument(
    '--stock_env',
    default='AAPL',
    metavar='ENV',
    help='stock environment to train on (default: AAPL)')
parser.add_argument(
    '--period_1',
    type=int,
    default=10,
    metavar='p1',
    help='main averaging period length of features')
parser.add_argument(
    '--period_2',
    type=int,
    default=25,
    metavar='p2',
    help='secondary averaging period length of features for MACD')
parser.add_argument(
    '--shared-optimizer',
    default=True,
    metavar='SO',
    help='use an optimizer without shared statistics.')
parser.add_argument(
    '--load', default=False, metavar='L', help='load a trained model')
parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')
parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='shares optimizer choice of Adam or RMSprop')
parser.add_argument(
    '--load_model_path',
    default='trained_models/',
    metavar='LMP',
    help='path to load trained model')
parser.add_argument(
    '--save_model_dir',
    default='trained_models/',
    metavar='SMD',
    help='folder to save trained models')
parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--amsgrad',
    default=True,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')
parser.add_argument(
    '--input_dim',
    type=int,
    default=11,
    help='state input dimension')
parser.add_argument(
    '--rl_input_dim',
    type=int,
    default=18,
    help='state input dimension')
parser.add_argument(
    '--num_actions',
    type=int,
    default=7,
    help='number of actions')


def load_dataset(stock_env, p1, p2):
    train_head_name = stock_env + "_train_data" + "_p1" + str(p1) + "_p2" + str(p2)
    train_norm_data = np.load(DATA_PATH + train_head_name + "_normalized.npy")
    train_raw_data = np.load(DATA_PATH + train_head_name + "_raw.npy")
    test_head_name = stock_env + "_test_data" + "_p1" + str(p1) + "_p2" + str(p2)
    test_norm_data = np.load(DATA_PATH + test_head_name + "_normalized.npy")
    test_raw_data = np.load(DATA_PATH + test_head_name + "_raw.npy")
    return train_norm_data, train_raw_data, test_norm_data, test_raw_data

def get_model_name(args):
    model_name = args.stock_env + "_p1" + str(args.period_1) + "_p2" + str(args.period_2) + "_minEL" + str(args.min_episode_length) + "_maxEL" + str(args.max_episode_length) + "_nstep" + str(args.num_steps) + "_lr" + str(args.lr) + "_gamma" + str(args.gamma) + "_tau" + str(args.tau)
    return model_name

def find_contiguous_colors(colors):
    # finds the continuous segments of colors and returns those segments
    segs = []
    curr_seg = []
    prev_color = ''
    for c in colors:
        if c == prev_color or prev_color == '':
            curr_seg.append(c)
        else:
            segs.append(curr_seg)
            curr_seg = []
            curr_seg.append(c)
        prev_color = c
    segs.append(curr_seg) # the final one
    return segs

def plot_multicolored_lines(x,y,colors):
    segments = find_contiguous_colors(colors)
    plt.figure()
    start= 0
    for seg in segments:
        end = start + len(seg)
        l, = plt.gca().plot(x[start:end],y[start:end],lw=2,c=seg[0])
        start = end

if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn')
    # Create environment dictionary
    starting_capital = 100000
    trans_cost_rate = 0.0005
    slippage_rate = 0.001
    max_position_per_act = 3
    train_norm_data, train_raw_data, test_norm_data, test_raw_data = load_dataset(args.stock_env, args.period_1, args.period_2)

    train_env_config = {}
    test_env_config = {}
    train_env_config['stock_raw_data'] = train_raw_data
    train_env_config['stock_norm_data'] = train_norm_data
    train_env_config['starting_capital'] = starting_capital
    train_env_config['trans_cost_rate'] = trans_cost_rate
    train_env_config['slippage_rate'] = slippage_rate
    train_env_config['min_episode_length'] = args.min_episode_length
    train_env_config['max_episode_length'] = args.max_episode_length
    train_env_config['max_position'] = (args.num_actions - 1) / 2
    test_env_config['stock_raw_data'] = test_raw_data
    test_env_config['stock_norm_data'] = test_norm_data
    test_env_config['starting_capital'] = starting_capital
    test_env_config['trans_cost_rate'] = trans_cost_rate
    test_env_config['slippage_rate'] = slippage_rate
    test_env_config['min_episode_length'] = args.min_episode_length
    test_env_config['max_episode_length'] = args.max_episode_length
    test_env_config['max_position'] = (args.num_actions - 1) / 2

    # Initiate and Load sdae models (TODO: add argument to specify file name)
    sdae_model_name = "AAL_sdae_model_lr4_g_noise_var0.001_pre100000fine500000.pt"
    sdae_model = SDAE(args.input_dim)
    sdae_saved_state = torch.load(SDAE_PATH + sdae_model_name, map_location=lambda storage, loc: storage)
    sdae_model.load_state_dict(sdae_saved_state)

    # Initiate shared model
    trained_model = A3C_LSTM(args.rl_input_dim, args.num_actions)
    model_name = get_model_name(args)
    #saved_state = torch.load('{0}{1}.pt'.format(args.load_model_path, model_name), map_location=lambda storage, loc: storage)
    saved_state = torch.load('{0}{1}.pt.dat'.format(args.load_model_path, model_name), map_location=lambda storage, loc: storage)
    trained_model.load_state_dict(saved_state)
    trained_model.share_memory()

    # Run one testing episode
    episodic_reward, rewards, actions = test_one_episode(args, sdae_model, trained_model, test_env_config)
    actions = [a - max_position_per_act for a in actions]
    colors = []
    red_count = 0
    blue_count = 0
    green_count = 0
    for a in actions:
        if(a < 0):
            colors.append('red')
            red_count += 1
        elif(a == 0):
            colors.append('blue')
            blue_count += 1
        else:
            colors.append('green')
            green_count += 1

    #print("red: " + str(red_count) + " green: " + str(green_count) + " blue: " + str(blue_count))
    #exit()

    # Plotting, change buy and sell actions to green and red
    test_indices = []
    test_prices = []
    for i in range(test_raw_data.shape[0]):
        test_indices.append(i)
        test_prices.append(test_raw_data[i, 6])
    plot_multicolored_lines(test_indices[1:-1], test_prices[1:-1], colors)
    plt.show()
