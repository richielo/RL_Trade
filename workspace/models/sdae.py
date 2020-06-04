import os
import sys
import numpy as np
import torch
from torch import nn
from torch.functional import F

# Model class for stacked denoising autoencoder
class SDAE(nn.module):
    def __init__(self, input_dim):
        # Encoder layers
        self.enc_fc_1 = nn.Linear(input_dim, 128)
        self.enc_fc_2 = nn.Linear(128, 128)
        self.enc_fc_3 = nn.Linear(128, 64)
        self.enc_fc_4 = nn.Linear(64, 16)
        self.encoder_object = nn.sequential(*[self.enc_fc_1, nn.ReLU(), self.enc_fc_2, nn.ReLU(), self.enc_fc_3, nn.ReLU(), self.enc_fc_4])

        # Decoder layers
        self.dec_fc_1 = nn.Linear(16, 64)
        self.dec_fc_2 = nn.Linear(64, 128)
        self.dec_fc_3 = nn.Linear(128, 128)
        self.dec_fc_4 = nn.Linear(128, input_dim)
        self.decoder_object = nn.sequential(*[self.dec_fc_1, nn.ReLU(), self.dec_fc_2, nn.ReLU(), self.dec_fc_3, nn.ReLU(), self.dec_fc_4])

    def forward(self, input, training = True):
        if(training):
            encoded_output = self.encoder_object(input)
            decoded_output = self.decoder_object(encoded_output)
            return decoded_output
        else:
            with torch.no_grad():
                encoded_output = self.encoder_object(input)
                return encoded_output
