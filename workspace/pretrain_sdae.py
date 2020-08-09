import os
import sys
import argparse
import time
import numpy as np
import torch.optim as optim
import torch
from torch import nn
from torch.functional import F
from models.sdae import *

EVAL_FREQ = 1000
BATCH_SIZE = 256
MODELS_PATH = "sdae_models/"

def pretraining(train_path, test_path, lr_index, g_noise_var, pre_num_iter, fine_num_iter, use_filter_data, filter_by_year):
    stock_name = train_path.split('/')[1].split('_')[0]
    p_1 = train_path.split('/')[1].split('_')[3].replace('p1', '')
    p_2 = train_path.split('/')[1].split('_')[4].replace('p2', '')
    # Load data
    # Remove timestamp
    train_data = np.load(train_path)[:, :11]
    test_data = np.load(test_path)[:, :11]
    train_data = torch.tensor(train_data).float()
    test_data = torch.tensor(test_data).float()

    # Initialize model and optimizer
    sdae_model = SDAE(11)
    model_optimizer = optim.Adam(sdae_model.parameters(), lr = 10 ** (-1.0 * lr_index))

    # Layer wise pretraining
    for layer_index in range(sdae_model.num_layers):
        sdae_model.freeze_all_but(layer_index)
        for iter in range(pre_num_iter):
            sample_indices = np.random.randint(train_data.shape[0], size=(BATCH_SIZE,))
            batch_input = train_data[sample_indices]
            batch_output = batch_input.clone()
            # Add gaussian noise
            corrupted_batch_input = batch_input + (g_noise_var ** 0.5) * torch.randn(batch_input.size())
            # Learning
            pred = sdae_model.forward(batch_input)
            loss = F.mse_loss(pred, batch_output)
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
        sdae_model.unfreeze_all()

    # Fine-tuning
    sdae_model.unfreeze_all()
    for iter in range(fine_num_iter):
        sample_indices = np.random.randint(train_data.shape[0], size=(BATCH_SIZE,))
        batch_input = train_data[sample_indices]
        batch_output = batch_input.clone()
        # Add gaussian noise
        corrupted_batch_input = batch_input + (g_noise_var ** 0.5) * torch.randn(batch_input.size())
        # Learning
        pred = sdae_model.forward(batch_input)
        loss = F.mse_loss(pred, batch_output)
        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()
        # Testing
        if((iter + 1) % EVAL_FREQ == 0):
            corrupted_test_input = test_data + (g_noise_var ** 0.5) * torch.randn(test_data.size())
            with torch.no_grad():
                pred = sdae_model.forward(corrupted_test_input)
                loss = F.mse_loss(pred, test_data)
                print("fine-tune iter: " + str(iter) + " loss: " + str(loss))
                sys.stdout.flush()

    if(use_filter_data):
        torch.save(sdae_model.state_dict(), MODELS_PATH + stock_name + "_p1" + str(p_1) + "_p2" + str(p_2) + "_sdae_model_lr" + str(lr_index) + "_g_noise_var" + str(g_noise_var) + "_pre" + str(pre_num_iter) + "fine" + str(fine_num_iter) + "_filtered_fyear" + str(filter_by_year) + ".pt")
    else:
        torch.save(sdae_model.state_dict(), MODELS_PATH + stock_name + "_p1" + str(p_1) + "_p2" + str(p_2) + "_sdae_model_lr" + str(lr_index) + "_g_noise_var" + str(g_noise_var) + "_pre" + str(pre_num_iter) + "fine" + str(fine_num_iter) + ".pt")

def main():
    parser = argparse.ArgumentParser()
    # Train data path
    parser.add_argument("--train_path", type = str, default = None)
    # Test data path
    parser.add_argument("--test_path", type = str, default = None)
    # learning rate index (negative powers of 10)
    parser.add_argument("--lr_index", type = int, default = 4)
    # Gaussian noise variance
    parser.add_argument("--g_noise_var", type = float, default = 0.001)
    # Layer-wise pretraining - Number of iterations
    parser.add_argument("--pre_num_iter", type = int, default = 100000)
    # Fine-tuning - Number of iterations
    parser.add_argument("--fine_num_iter", type = int, default = 1000000)
    # Use filter data or not
    parser.add_argument('--use_filter_data', default=True, help='Whether to use filtered data')
    # Filter by year
    parser.add_argument('--filter_by_year', type = int, default=2000, help='The oldest year to include')
    args = parser.parse_args()

    pretraining(args.train_path, args.test_path, args.lr_index, args.g_noise_var, args.pre_num_iter, args.fine_num_iter, args.use_filter_data, args.filter_by_year)

if __name__ == '__main__':
	main()
