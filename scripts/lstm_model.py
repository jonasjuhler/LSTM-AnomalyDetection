import torch.nn as nn
import torch
import numpy as np

class LSTM_VAE(nn.Module):
    def __init__(self, z_dim, T, x_dim, h_dim, batch_size=1):
        super(LSTM_VAE, self).__init__()

        self.z_dim = z_dim
        self.T = T
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.batch_size = batch_size
        
        # Everything going on in the network has to be of size:
        # (batch_size, T, n_features)
        
        # We encode the data onto the latent space using bi-directional LSTM
        self.encoder = nn.LSTM(
            input_size=self.x_dim,
            hidden_size=self.h_dim,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )
        
        self.fnn_zmu = nn.Linear(
            in_features=2*self.h_dim,
            out_features=self.z_dim,
            bias=True
        )

        self.fnn_zvar = nn.Linear(
            in_features=2*self.h_dim,
            out_features=self.z_dim,
            bias=True
        )
        
        self.fnn_cmu = nn.Linear(
            in_features=2*self.h_dim,
            out_features=self.z_dim,
            bias=True
        )
        
        self.fnn_cvar = nn.Linear(
            in_features=2*self.h_dim,
            out_features=self.z_dim,
            bias=True
        )
        
        # The latent code must be decoded into the original image
        self.decoder = nn.LSTM(
            input_size=self.z_dim*2,
            hidden_size=self.h_dim,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )
        
        self.fnn_xmu = nn.Linear(
            in_features=2*self.h_dim,
            out_features=self.x_dim,
            bias=True
        )
        
        self.fnn_xb = nn.Linear(
            in_features=2*self.h_dim,
            out_features=self.x_dim,
            bias=True
        ) 

        # Initialize hidden state and cell state as learnable parameters
        self.hidden_state = torch.zeros(2, 1, self.h_dim)
        self.cell_state = torch.zeros(2, 1, self.h_dim)

        
    def forward(self, x):
        outputs = {}
        
        out_encoded, (hidden_T, _) = self.encoder(x, (self.hidden_state, self.cell_state))
        flat_hidden_T = hidden_T.reshape(1, 1, 2*self.h_dim)
        
        # Fully connected layer from LSTM to var and mu
        mu_z = self.fnn_zmu(flat_hidden_T)
        sigma_z = nn.functional.softplus(self.fnn_zvar(flat_hidden_T))

        # :- Reparametrisation trick
        # a sample from N(mu, sigma) is mu + sigma * epsilon
        # where epsilon ~ N(0, 1)

        # Don't propagate gradients through randomness
        with torch.no_grad():
            epsilon_z = torch.randn(
                self.batch_size, 1, self.z_dim)

        # (batch_size, latent_dim) -> (batch_size, 1, latent_dim)
        z = mu_z + epsilon_z * sigma_z        

        # Here goes the Variational Self-Attention Network 
        
        # Calculate the similarity matrix
        S = torch.zeros(self.batch_size, self.T, self.T)
        for b in range(self.batch_size):
            for i in range(self.T):
                for j in range(i, self.T):
                    S[b,i,j] = torch.dot(out_encoded[b,i,:], out_encoded[b,j,:])
                    S[b,j,i] = S[b,i,j]
        
        S = S / np.sqrt((2 * self.h_dim))
        
        # Use softmax to get the sum of weights to equal 1
        A = nn.functional.softmax(S, dim=2)
        
        Cdet = torch.zeros(self.batch_size, self.T, 2*self.h_dim)
        
        for b in range(self.batch_size):
            Cdet[b] = torch.matmul(A[b], out_encoded[b])
            
            
        # Fully connected layer from LSTM to var and mu
        mu_c = self.fnn_cmu(Cdet)
        sigma_c = nn.functional.softplus(self.fnn_cvar(Cdet))

        # Don't propagate gradients through randomness
        with torch.no_grad():
            epsilon_c = torch.randn(
                self.batch_size, self.T, self.z_dim)

        c = mu_c + epsilon_c * sigma_c     
        
        # Concatenate z and c before giving it as input to the decoder
        z = torch.cat(self.T*[z], dim=1)
        zc_concat = torch.cat((z,c), dim=2)
        
        # Run through decoder
        out_decoded, _ = self.decoder(zc_concat)
        
        # Pass the decoder outputs through fnn to get LaPlace parameters
        mu_x = self.fnn_xmu(out_decoded)
        b_x = nn.functional.softplus(self.fnn_xb(out_decoded))
        
        outputs["z"] = z
        outputs["mu_z"] = mu_z
        outputs["sigma_z"] = sigma_z
        outputs["c"] = c
        outputs["mu_c"] = mu_c
        outputs["sigma_c"] = sigma_c
        outputs["mu_x"] = mu_x
        outputs["b_x"] = b_x
        
        return outputs

import os
os.getcwd()
from lstmAD.dataload.load_data import load_data

train, _ = load_data('ItalyPowerDemand', basepath='..')

train_x = torch.Tensor(train[:5, 20]).unsqueeze(1).unsqueeze(0)

print(train_x.shape)

z_dim = 3
N = train_x.shape[0]
T = train_x.shape[1]
x_dim = train_x.shape[2]
h_dim = 4

net = LSTM_VAE(z_dim, T, x_dim, h_dim)

output = net(train_x)
