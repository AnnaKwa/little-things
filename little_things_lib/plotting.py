import matplotlib.pyplot as plt
import numpy as np


def plot_posterior_distributions(
        sampler,
        labels= ["log10(rho0)", "log10(sigma0)", "log10(cross)", "ML disk"]
):
    assert len(labels)==sampler.chain.shape[2], \
        "Number of labels must equal number of parameters in sampler."
    fig = plt.figure(figsize=(5, 4*len(labels)))
    for i in range(sampler.chain.shape[2]):
        ax = fig.add_subplot(len(labels), 1, i+1)
        ax.hist(sampler.chain[:, :, i].flatten())
        ax.set_xlabel(labels[i])
    plt.show()


def plot_walker_paths(
        sampler,
        mcmc_params,
        labels=["log10(rho0)", "log10(sigma0)", "log10(cross)", "ML disk"]
):
    fig, ax = plt.subplots(len(labels), sharex = True, figsize=(10,5*len(labels)))

    for i,iplot in zip(range(4),range(len(labels))):
        for j in range(mcmc_params.nwalkers):
            ax[iplot].plot(sampler.chain[j,:,i])
        ax[iplot].set_ylabel(labels[i])
        ax[iplot].set_xlabel('step')
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)