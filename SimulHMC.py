import numpy as np
import matplotlib.pyplot as plt
from func import *


def SimulMC(I, t, N, ch):

    X = np.zeros(shape=(N), dtype='uint')
    # simulation of the first
    X[0] = np.random.choice(np.size(I), 1, p=I)
    # simulation of the following
    for n in range(N-1):
        X[n+1] = np.random.choice(np.size(I), 1, p=t[X[n], :])

    # plot
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np.arange(N-60, N), X[N-60:N], lw=2, alpha=0.9, color='b')
    plt.savefig('./results/SimulHMC_X_' + str(N) + ch +
                '.png', bbox_inches='tight', dpi=150)

    return X


def SimulObs(mu, sigma, X, ch):

    # simulation of the first
    Y = np.zeros(shape=(len(X)))
    # simulation of the following
    for n in range(N):
        Y[n] = np.random.normal(loc=mu[X[n]], scale=sigma[X[n]], size=1)

    # plot
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np.arange(N-60, N), Y[N-60:N], lw=2, alpha=0.9, color='b')
    plt.savefig('./results/SimulHMC_Y_' + str(N) + ch +
                '.png', bbox_inches='tight', dpi=150)

    # Histograms
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # binwidth = 0.15
    # lim = np.ceil(np.max(np.abs(Y))/1.5 / binwidth) * binwidth
    # ax.set_xlim((-lim, lim))
    # bins = np.arange(-lim, lim + binwidth, binwidth)
    # ax.hist(Y, bins=bins, density=True)
    # plt.savefig('./results/SimulHMC_HistoY_' + str(N) + ch + '.png', bbox_inches='tight', dpi=150)

    return Y


if __name__ == '__main__':

    # Nombre d'échantillons à simuler
    N = 1000

    # Gaussians
    # mu    = [0., 2.0, 4.0]
    # sigma = [1., 2., 1.]
    mu = [100, 110]
    sigma = [6, 3]
    K = np.shape(mu)[0]  # le nombre de classes

    # Homogeneous and stationary Markov chain
    t2, ch2 = np.array(
        [[0.7, 0.1, 0.2], [0.3, 0.6, 0.1], [0.2, 0.3, 0.5]]), "_t2"
    t3, ch3 = np.array(
        [[0.2, 0.4, 0.4], [0.4, 0.3, 0.3], [0.3, 0.5, 0.2]]), "_t3"
    t4, ch4 = np.array([[0.95, 0.05], [0.05, 0.95]]), "_t4"

    t, ch = t4, ch4
    I = getSteadyState(t)
    print('I=', I, '\nt=', t)

    # Simulation de la chaine de Markov
    X = SimulMC(I, t, N, ch)

    # Simulation des observations selon le modèle des HMC
    Y = SimulObs(mu, sigma, X, ch)

    # Save generated signals
    np.savetxt('sources/XY'+ch+'.out', (X, Y))
