import torch
import torch.nn as nn
import numpy as np


class VRASAM(nn.Module):
    def __init__(self, z_dim, T, x_dim, h_dim, batch_size=1):
        super(VRASAM, self).__init__()

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
        self.hidden_state = torch.nn.Parameter(torch.zeros(2, 1, self.h_dim))
        self.cell_state = torch.nn.Parameter(torch.zeros(2, 1, self.h_dim))

    def forward(self, x):
        # TODO: Add monte-carlo integration
        outputs = {}

        out_encoded, (hidden_T, _) = self.encoder(
            x, (self.hidden_state, self.cell_state))
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
            S[b] = torch.matmul(
                out_encoded[b],
                torch.transpose(out_encoded[b], 0, 1)
            )

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
        z_cat = torch.cat(self.T*[z], dim=1)
        zc_concat = torch.cat((z_cat, c), dim=2)

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


def ELBO_loss(x, outputs, lambda_KL=1, eta_KL=1):
    mu_x = outputs['mu_x']
    b_x = outputs['b_x']
    mu_z = outputs["mu_z"]
    sigma_z = outputs["sigma_z"]
    mu_c = outputs["mu_c"]
    sigma_c = outputs["sigma_c"]

    pdf_laplace = torch.distributions.laplace.Laplace(mu_x, b_x)
    likelihood = pdf_laplace.log_prob(x).mean(dim=1)

    kl_z = -0.5 * torch.sum(1 + torch.log(sigma_z**2) -
                            mu_z**2 - sigma_z**2, dim=2)
    kl_c = -0.5 * torch.sum(1 + torch.log(sigma_c**2) -
                            mu_c**2 - sigma_c**2, dim=2)
    kl_c = torch.mean(kl_c, dim=1)

    ELBO = -torch.mean(likelihood) + lambda_KL*kl_z + eta_KL*kl_c

    return ELBO


def similarity_score():
    pass
