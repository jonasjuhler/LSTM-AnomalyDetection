import numpy as np
import torch
from kongkat.dataload.generate_data import gendata
from kongkat.model.vrasam import similarity_score


def results_plot(ax, net, x_normal, x_outlier, L=1):

    # Set network batch size down and reinitalize hidden- and cell state
    net.batch_size = 1
    net.init_hidden()

    # Define range of plotting
    plot_lim = (-0.1, 1)

    # Create regeneration
    outputs_normal = net(x_normal)
    outputs_outlier = net(x_outlier)
    mu_x_normal = outputs_normal['mu_x']
    mu_x_outlier = outputs_outlier['mu_x']

    # Calculate similarity
    sim_normal = similarity_score(net, x_normal, L=L)
    sim_outlier = similarity_score(net, x_outlier, L=L)

    # Plot ground truth
    ax[0, 0].plot(x_normal[0])
    ax[0, 0].set_ylim(plot_lim)
    ax[0, 0].set_title('Ground truth')

    # Plot regenerated ground truth
    ax[0, 1].plot(mu_x_normal.detach().numpy()[0], 'r-', label='Regenerated')
    ax[0, 1].set_title('Regenerated ground truth')
    ax[0, 1].set_ylim(plot_lim)
    ax_sim = ax[0, 1].twinx()
    ax_sim.plot(sim_normal.detach().numpy()[0], 'k-', label='Similarity')
    ax[0, 1].legend(fontsize=8, loc='upper left')
    ax_sim.legend(fontsize=8, loc='upper right')

    # Plot outlier data
    ax[1, 0].plot(x_outlier[0])
    ax[1, 0].set_ylim(plot_lim)
    ax[1, 0].set_title('Outlier timeseries')

    # Plot regenerated outlier data
    ax[1, 1].plot(mu_x_outlier.detach().numpy()[0], 'r-', label='Regenerated')
    ax[1, 1].set_title('Regenerated outlier data')
    ax[1, 1].set_ylim(plot_lim)
    ax_sim = ax[1, 1].twinx()
    ax_sim.plot(sim_outlier.detach().numpy()[0], 'k-', label='Similarity')
    ax[1, 1].legend(fontsize=8, loc='upper left')
    ax_sim.legend(fontsize=8, loc='upper right')

    return ax
