
import torch
import numpy as np
import matplotlib.pyplot as plt
from kongkat.dataload.generate_data import gendata
from kongkat.model.vrasam import VRASAM

anomaly = "Fault"

ts_norm = gendata()
ts_out, anom_range = gendata(outlier_type=anomaly)

ts_normal = np.concatenate(2*[ts_norm])
ts_outlier = np.concatenate([ts_norm]+[ts_out])


x_normal = torch.Tensor(ts_normal).unsqueeze(1).unsqueeze(0)
x_outlier = torch.Tensor(ts_outlier).unsqueeze(1).unsqueeze(0)
print("Data dim: (batch_size={0}, T={1}, x_dim={2})".format(*x_normal.shape))

z_dim = 3
batch_size, T, x_dim = x_normal.shape
h_dim = 4

print("Initializing model with specs:")
print("Batch size:", batch_size)
print("Dimension of LSTM hidden state (encoder and decoder):", h_dim)
print("Dimension of latent space variables z and c's:", z_dim)
net = VRASAM(z_dim, T, x_dim, h_dim)

# Make forward pass with model
output_normal = net(x_normal)
output_outlier = net(x_outlier)

# Extract the Laplace distribution parameters from model.
mu_x_normal = output_normal["mu_x"].detach().numpy()
b_x_normal = output_normal["b_x"].detach().numpy()
mu_x_outlier = output_outlier["mu_x"].detach().numpy()
b_x_outlier = output_outlier["b_x"].detach().numpy()


# Sample reconstructed x from the given parameters
x_hat_normal = np.random.laplace(loc=mu_x_normal, scale=b_x_normal)
x_hat_outlier = np.random.laplace(loc=mu_x_outlier, scale=b_x_outlier)

fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(x_normal[0])
ax[1, 0].plot(x_hat_normal[0])
ax[0, 1].plot(x_outlier[0])
ax[1, 1].plot(x_hat_outlier[0])
ax[0, 0].set_ylim((0, 1))
ax[0, 1].set_ylim((0, 1))
ax[0, 0].set_title('Ground truth')
ax[0, 1].set_title(anomaly)
ax[1, 0].set_title('Regenerated ground truth')
ax[1, 1].set_title(f'Regenerated {anomaly.lower()}')

plt.show()
