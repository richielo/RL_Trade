import os
import sys
import numpy as np
import torch
from torch import nn
from torch.functional import F

# Model class for stacked denoising autoencoder
class SDAE(nn.Module):
    def __init__(self, input_dim):
        super(SDAE, self).__init__()
        self.num_layers = 4

        # Paper version
        # Encoder layers
        self.enc_fc_1 = nn.Linear(input_dim, 10)
        self.enc_fc_2 = nn.Linear(10, 16)
        self.encoder_object = nn.Sequential(*[self.enc_fc_1, nn.ReLU(), self.enc_fc_2])
        # Decoder layers
        self.dec_fc_1 = nn.Linear(16, 10)
        self.dec_fc_2 = nn.Linear(10, input_dim)
        self.decoder_object = nn.Sequential(*[self.dec_fc_1, nn.ReLU(), self.dec_fc_2])

        """
        # Old version - misinterpretation of the paper
        # Encoder layers
        self.enc_fc_1 = nn.Linear(input_dim, 128)
        self.enc_fc_2 = nn.Linear(128, 128)
        self.enc_fc_3 = nn.Linear(128, 64)
        self.enc_fc_4 = nn.Linear(64, 16)
        self.encoder_object = nn.Sequential(*[self.enc_fc_1, nn.ReLU(), self.enc_fc_2, nn.ReLU(), self.enc_fc_3, nn.ReLU(), self.enc_fc_4])

        # Decoder layers
        self.dec_fc_1 = nn.Linear(16, 64)
        self.dec_fc_2 = nn.Linear(64, 128)
        self.dec_fc_3 = nn.Linear(128, 128)
        self.dec_fc_4 = nn.Linear(128, input_dim)
        self.decoder_object = nn.Sequential(*[self.dec_fc_1, nn.ReLU(), self.dec_fc_2, nn.ReLU(), self.dec_fc_3, nn.ReLU(), self.dec_fc_4])
        """

    def forward(self, input, training = True):
        if(training):
            encoded_output = self.encoder_object(input)
            decoded_output = self.decoder_object(encoded_output)
            return decoded_output
        else:
            with torch.no_grad():
                encoded_output = self.encoder_object(input)
                return encoded_output

    def freeze_all_but(self, layer_index):
        # This function freezes everything but the indicated layer, used for pretraining
        if(layer_index >= self.num_layers):
            print("Layer index error")
            exit()
        layer_count = 0
        for name, param in self.encoder_object.named_parameters():
            if(layer_index != layer_count):
                param.requires_grad = False
            if('bias' in name):
                layer_count += 1
        layer_count = self.num_layers - 1
        for name, param in self.decoder_object.named_parameters():
            if(layer_index != layer_count):
                param.requires_grad = False
            if('bias' in name):
                layer_count -= 1

        # Check for freeze correctness
        """
        for name, param in self.encoder_object.named_parameters():
            print(param.requires_grad)
        for name, param in self.decoder_object.named_parameters():
            print(param.requires_grad)
        """

    def unfreeze_all(self):
        for name, param in self.encoder_object.named_parameters():
            param.requires_grad = True
        for name, param in self.decoder_object.named_parameters():
            param.requires_grad = True
