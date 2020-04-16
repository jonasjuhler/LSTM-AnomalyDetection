import torch
import numpy as np
import matplotlib.pyplot as plt
from kongkat.dataload.generate_data import gendata, data_generator
from kongkat.model.vrasam import VRASAM, ELBO_loss

T = 96  # Length of sequences
N = 800  # Number of sequences
T_w = 32  # Window length
N_w = int(T/T_w * N)  # Number of windows
batch_size = 1
x_dim = 1

X = torch.Tensor([X for X, _ in data_generator(N, T=T)]
                 ).view(N_w, batch_size, T_w, -1)
z_dim = 3
h_dim = 4

print("Initializing model with specs:")
print("Batch size:", batch_size)
print("Dimension of LSTM hidden state (encoder and decoder):", h_dim)
print("Dimension of latent space variables z and c's:", z_dim)
net = VRASAM(z_dim, T_w, x_dim, h_dim)
criterion = ELBO_loss
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

fig, ax = plt.subplots(1, 1)
plot_lim = (-0.1, 1)

true_line, = ax.plot(X[0, 0], 'b-', label='Ground truth')
ax.set_ylim(plot_lim)
ax.set_title('Timeseries')
regen_line, = ax.plot(range(T_w), np.random.rand(T_w),
                      'r-', label='Regenerated')
ax.set_ylim(plot_lim)

plt.ion()
plt.legend()
plt.show()


for i, x_i in enumerate(X):
    # Make forward pass with model
    optimizer.zero_grad()
    outputs = net(x_i)

    mu_x = outputs['mu_x']

    loss = criterion(x_i, outputs)
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        true_line.set_ydata(x_i.detach().numpy()[0])
        regen_line.set_ydata(mu_x.detach().numpy())
        plt.draw()
        plt.pause(0.001)
        print(loss)

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
