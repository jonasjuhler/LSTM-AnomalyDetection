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
        self.init_hidden()

        # Everything going on in the network has to be of size:
        # (batch_size, T, n_features)

        # We encode the data onto the latent space using bi-directional LSTM
        self.LSTM_encoder = nn.LSTM(
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
        self.LSTM_decoder = nn.LSTM(
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

    def init_hidden(self):
        # Initialize hidden state and cell state as learnable parameters
        self.hidden_state = torch.zeros(2, self.batch_size, self.h_dim)
        self.cell_state = torch.zeros(2, self.batch_size, self.h_dim)

    def encoder(self, x):
        out_encoded, (hidden_T, _) = self.LSTM_encoder(
            x, (self.hidden_state, self.cell_state))
        flat_hidden_T = hidden_T.reshape(self.batch_size, 1, 2*self.h_dim)

        # Fully connected layer from LSTM to var and mu
        mu_z = self.fnn_zmu(flat_hidden_T)
        sigma_z = nn.functional.softplus(self.fnn_zvar(flat_hidden_T))

        # Calculate the similarity matrix
        S = torch.matmul(
            out_encoded,
            torch.transpose(out_encoded, 1, 2)
        )

        S = S / np.sqrt((2 * self.h_dim))

        # Use softmax to get the sum of weights to equal 1
        A = nn.functional.softmax(S, dim=2)
        Cdet = torch.matmul(A, out_encoded)

        # Fully connected layer from LSTM to var and mu
        mu_c = self.fnn_cmu(Cdet)
        sigma_c = nn.functional.softplus(self.fnn_cvar(Cdet))

        return mu_z, sigma_z, mu_c, sigma_c

    def decoder(self, c, z):
        # Concatenate z and c before giving it as input to the decoder
        z_cat = torch.cat(self.T*[z], dim=1)
        zc_concat = torch.cat((z_cat, c), dim=2)

        # Run through decoder
        out_decoded, _ = self.LSTM_decoder(zc_concat)

        # Pass the decoder outputs through fnn to get LaPlace parameters
        mu_x = self.fnn_xmu(out_decoded)
        b_x = nn.functional.softplus(self.fnn_xb(out_decoded))

        return mu_x, b_x

    def forward(self, x):
        # TODO: Add monte-carlo integration
        outputs = {}

        mu_z, sigma_z, mu_c, sigma_c = self.encoder(x)

        # Don't propagate gradients through randomness
        with torch.no_grad():
            epsilon_z = torch.randn(self.batch_size, 1, self.z_dim)
            epsilon_c = torch.randn(self.batch_size, self.T, self.z_dim)

        z = mu_z + epsilon_z * sigma_z
        c = mu_c + epsilon_c * sigma_c

        mu_x, b_x = self.decoder(c, z)

        outputs["z"] = z
        outputs["mu_z"] = mu_z
        outputs["sigma_z"] = sigma_z
        outputs["c"] = c
        outputs["mu_c"] = mu_c
        outputs["sigma_c"] = sigma_c
        outputs["mu_x"] = mu_x
        outputs["b_x"] = b_x

        return outputs

    def count_parameters(self):
        n_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        return n_grad, n_total


def ELBO_loss(x, outputs, lambda_KL=0.01, eta_KL=0.01):
    mu_x = outputs['mu_x']
    b_x = outputs['b_x']
    mu_z = outputs["mu_z"]
    sigma_z = outputs["sigma_z"]
    mu_c = outputs["mu_c"]
    sigma_c = outputs["sigma_c"]

    # Initialize Laplace with given parameters.
    pdf_laplace = torch.distributions.laplace.Laplace(mu_x, b_x)
    # Calculate mean of likelihood over T and x-dimension.
    likelihood = pdf_laplace.log_prob(x).mean(dim=1).mean(dim=1)

    # Calculate KL-divergence of p(c) and q(z)
    v_z = sigma_z**2  # Variance of q(z)
    v_c = sigma_c**2  # Variance of p(c)
    kl_z = -0.5 * torch.sum(1 + torch.log(v_z) - mu_z**2 - v_z, dim=2)
    kl_c = -0.5 * torch.sum(1 + torch.log(v_c) - mu_c**2 - v_c, dim=2)

    kl_c = torch.sum(kl_c, dim=1)  # Sum over the T dimension.
    kl_z = kl_z[:, 0]  # Get rid of extra x_dim dimension.

    # Calculate the Evidence Lower Bound (ELBO)
    ELBO = -likelihood + (lambda_KL * kl_z) + (eta_KL * kl_c)

    # Return mean over all batches
    return torch.mean(ELBO, dim=0)


def similarity_score(net, x, L):

    with torch.no_grad():
        # Pass sequence through encoder to get params in q(z) and p(c)
        mu_z, sigma_z, mu_c, sigma_c = net.encoder(x)
        score = 0

        for _ in range(L):
            # Sample a random z vector and reparametrize
            epsilon_z = torch.randn(net.batch_size, 1, net.z_dim)
            z = mu_z + epsilon_z * sigma_z

            # Pass sample through decoder and calculate reconstruction prob
            mu_x, b_x = net.decoder(mu_c, z)
            pdf_laplace = torch.distributions.laplace.Laplace(mu_x, b_x)
            score += pdf_laplace.log_prob(x)

        # Average over number of iterations
        return score/L
