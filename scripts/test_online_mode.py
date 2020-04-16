import torch
import numpy as np
import matplotlib.pyplot as plt
from kongkat.dataload.generate_data import gendata, data_generator
from kongkat.model.vrasam import VRASAM, ELBO_loss

T = 96  # Length of sequences
N = 100  # Number of sequences
T_w = 32  # Window length
N_w = T - T_w + 1  # Number of windows for each sequence
batch_size = 1
x_dim = 1
data_shape = (N, batch_size, T, x_dim)
gen = data_generator(N, T=T)  # Generate N samples of length T
X = torch.Tensor([X for X, _ in gen]).view(*data_shape)

# Model hyperparameters
z_dim = 3
h_dim = 128  # Number of LSTM units in each direction
lr = 0.001
criterion = ELBO_loss
net = VRASAM(z_dim, T_w, x_dim, h_dim)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
print("Initializing model with specs:")
print("Batch size:", batch_size)
print("Dimension of LSTM hidden state (encoder and decoder):", h_dim)
print("Dimension of latent space variables z and c's:", z_dim)

fig, ax = plt.subplots(1, 1)
plot_lim = (-0.1, 1)

true_line, = ax.plot(range(T_w), np.random.rand(T_w),
                     'b-', label='Normal data')
ax.set_ylim(plot_lim)
ax.set_title('Timeseries')
regen_line, = ax.plot(range(T_w), np.random.rand(T_w),
                      'r-', label='Regenerated')
ax.set_ylim(plot_lim)

plt.ion()
plt.legend()
plt.show()

p = 0

for x_i in X:
    # Sliding window
    for w_start in range(N_w):
        # Extract window from x_i
        x = x_i[:, w_start:(w_start+T_w)]

        # Make forward pass with model
        optimizer.zero_grad()
        outputs = net(x)

        mu_x = outputs['mu_x']

        loss = criterion(x, outputs, 0.01, 0.01)
        loss.backward()
        optimizer.step()

        if p % 100 == 0:
            true_line.set_ydata(x.detach().numpy()[0])
            regen_line.set_ydata(mu_x.detach().numpy())
            plt.draw()
            plt.pause(0.001)
            print(loss)

        p += 1
plt.close()

# Define normal and outlier type data
outlier_type = "Snow"
X_normal, _ = gendata(T=T_w)
X_outlier, _ = gendata(T=T_w, outlier_type=outlier_type)
X_normal = torch.Tensor(X_normal).unsqueeze(1).unsqueeze(0)
X_outlier = torch.Tensor(X_outlier).unsqueeze(1).unsqueeze(0)

# Create final regeneration
outputs_normal = net(X_normal)
outputs_outlier = net(X_outlier)
mu_x_normal = outputs_normal['mu_x']
mu_x_outlier = outputs_outlier['mu_x']


fig, ax = plt.subplots(2, 2, figsize=(10, 7))

# Plot ground truth
ax[0, 0].plot(X_normal[0])
ax[0, 0].set_ylim(plot_lim)
ax[0, 0].set_title('Ground truth')

# Plot regenerated ground truth
ax[0, 1].plot(mu_x_normal.detach().numpy()[0], 'r-')
ax[0, 1].set_title('Regenerated ground truth')
ax[0, 1].set_ylim(plot_lim)

# Plot outlier
ax[1, 0].plot(X_outlier[0])
ax[1, 0].set_ylim(plot_lim)
ax[1, 0].set_title('Outlier timeseries ({0})'.format(outlier_type.lower()))

# Plot regenerated ground truth
ax[1, 1].plot(mu_x_outlier.detach().numpy()[0], 'r-')
ax[1, 1].set_title('Regenerated outlier data')
ax[1, 1].set_ylim(plot_lim)

print("<Click figure to close>")
plt.waitforbuttonpress(0)  # this will wait for indefinite time
plt.close(fig)
