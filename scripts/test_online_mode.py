import torch
import numpy as np
import matplotlib.pyplot as plt
from kongkat.dataload.generate_data import gendata, data_generator
from kongkat.visualizations.model_plotting import results_plot
from kongkat.model.vrasam import VRASAM, ELBO_loss

T = 96  # Length of sequences
N = 10  # Number of sequences
T_w = 32  # Window length
N_w = T - T_w + 1  # Number of windows for each sequence
batch_size = 1
x_dim = 1
data_shape = (int(N/batch_size), batch_size, T, x_dim)
gen = data_generator(N, T=T)  # Generate N samples of length T
X = torch.Tensor([X for X, _ in gen]).view(*data_shape)

# Model hyperparameters
z_dim = 3
h_dim = 128  # Number of LSTM units in each direction
lr = 0.001
criterion = ELBO_loss
net = VRASAM(z_dim, T_w, x_dim, h_dim, batch_size)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, amsgrad=True)
print("Initializing model with specs:")
print("Batch size:", batch_size)
print("Dimension of LSTM hidden state (encoder and decoder):", h_dim)
print("Dimension of latent space variables z and c's:", z_dim)

fig, ax = plt.subplots(1, 1)
plot_lim = (-0.1, 1)

true_line, = ax.plot(range(T_w), range(T_w), 'b-', label='Normal data')
regen_line, = ax.plot(range(T_w), range(T_w), 'r-', label='Regenerated')

ax.set_title('Network output')
ax.set_ylim(plot_lim)

plt.ion()
plt.legend()

p = 0

for x_seq in X:
    # Sliding window loop
    for w_start in range(N_w):
        # Extract window from x_i
        x = x_seq[:, w_start:(w_start+T_w)]

        # Make forward pass with model
        optimizer.zero_grad()
        outputs = net(x)

        mu_x = outputs['mu_x']

        loss = criterion(x, outputs, 0.01, 0.01)
        loss.backward()
        optimizer.step()

        if p % 20 == 0:
            true_line.set_ydata(x.detach().numpy()[0])
            regen_line.set_ydata(mu_x.detach().numpy()[0])
            plt.draw()
            plt.pause(0.001)
            print(loss)

        p += 1

plt.close()

fig, ax = plt.subplots(2, 2, figsize=(10, 7))

# Visualize the reconstruction results on normal and outlier series
outlier_type = "Snow"
x_normal, _ = gendata(T=T)
x_outlier, _ = gendata(T=T, outlier_type=outlier_type)
# Extract sequence
start = np.random.randint(T - T_w + 1)  # Random starting point for seq
x_normal = x_normal[start:start+T_w]
x_outlier = x_outlier[start:start+T_w]
# Insert batch and x dim
x_normal = torch.Tensor(x_normal).view(1, -1, 1)
x_outlier = torch.Tensor(x_outlier).view(1, -1, 1)

ax = results_plot(ax, net, x_normal, x_outlier, L=1)

print("<Click figure to close>")
plt.waitforbuttonpress(0)  # this will wait for indefinite time
plt.close(fig)
