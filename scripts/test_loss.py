import torch
import numpy as np
import matplotlib.pyplot as plt
from kongkat.dataload.generate_data import gendata, data_generator
from kongkat.model.vrasam import VRASAM, ELBO_loss



def plot_fig(x, mu_x, b_x):
    # Sample reconstructed x from the given parameters
    x_gen = np.random.laplace(loc=mu_x.detach().numpy(), scale=b_x.detach().numpy())
    
    fig, ax = plt.subplots(1, 2)
    
    ax[0].plot(x[0])
    ax[1].plot(x_gen[0])

    ax[0].set_ylim((0, 1))
    ax[1].set_ylim((0, 1))
    ax[0].set_title('Ground truth')
    ax[1].set_title('Regenerated ground truth')
    
    plt.show()


X = 0.5*np.sin(np.arange(0,2*np.pi,0.4)) + 0.5
X = torch.Tensor(X).unsqueeze(1).unsqueeze(0)
z_dim = 3
batch_size, T, x_dim = X.shape
h_dim = 4

print("Initializing model with specs:")
print("Batch size:", batch_size)
print("Dimension of LSTM hidden state (encoder and decoder):", h_dim)
print("Dimension of latent space variables z and c's:", z_dim)
net = VRASAM(z_dim, T, x_dim, h_dim)
criterion = ELBO_loss
optimizer = torch.optim.Adam(net.parameters())

for i in range(2000):
    # Make forward pass with model
    #X = torch.Tensor(X).unsqueeze(1).unsqueeze(0)
    optimizer.zero_grad()
    outputs = net(X)
    
    mu_x = outputs['mu_x']
    b_x = outputs['b_x']
      
    loss = criterion(X, outputs)
    loss.backward()
    optimizer.step()
    
    
    if i%50 == 0:
        plot_fig(X, mu_x, b_x)
        print(loss)

    
    

