import torch


def train_online(net, X, T_w, epochs, optimizer, criterion):

    N, batch_size, T, x_dim = X.shape
    N_w = T - T_w + 1  # Number of windows for each sequence
    epoch_loss = torch.zeros(epochs)

    for e in range(epochs):

        e_loss = 0

        # Iterate through all sequences
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
                e_loss += loss.item()

        e_loss = e_loss / (N * N_w)
        print(
            "Epoch {0}/{1} done! - Average loss per sequence: {2:.2f}".format(
                e+1, epochs, e_loss)
        )
        epoch_loss[e] = e_loss

    return net
