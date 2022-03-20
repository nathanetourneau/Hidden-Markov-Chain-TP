import numpy as np
import matplotlib.pyplot as plt
from func import *


if __name__ == '__main__':

    #############################################@
    ####### Set the parameters and read the data

    # Gaussians
    # mu    = [0., 2.0, 4.0]
    # sigma = [1., 2., 1.]
    mu    = [100, 110]
    sigma = [6, 3]
    K = np.shape(mu)[0]  # le nombre de classes

    # Homogeneous and stationary Markov chain
    # t2, ch2 = np.array([[0.7, 0.1, 0.2], [0.3, 0.6, 0.1], [0.2, 0.3, 0.5]]), "_t2"
    # t3, ch3 = np.array([[0.2, 0.4, 0.4], [0.4, 0.3, 0.3], [0.3, 0.5, 0.2]]), "_t3"
    t4, ch4 = np.array([[0.95, 0.05], [0.05, 0.95]]), "_t4"

    t, ch = t4, ch4
    I = getSteadyState(t)
    print('I=', I, '\nt=', t)
    
    # The data (simulated)
    X, Y = np.loadtxt('sources/XY' + ch + '.out')
    N    = X.shape[0]

    #############################################@
    ####### MPM Restoration
    
    # forward computing
    alpha, S = getAlpha(K, N, Y, mu, sigma, I, t)
    
    # backward computing
    beta = getBeta(K, N, Y, mu, sigma, I, t, S)
    
    # gamma computing (marginal a posterori proba)
    gamma = getGamma(K, N, alpha, beta)

    # MPM classification
    X_MPM = getMPMClassif(N, gamma)

    # error rate compuation
    ConfMatrix_MPM, ERGlobal_MPM, ERbyClass_MPM = getConfMat(K, N, X, X_MPM)
    print('Confusion matrix for MPM:\n', ConfMatrix_MPM)
    print('Global error rate for MPM:', ERGlobal_MPM)
    print('By class error rate for MPM:', ERbyClass_MPM)
    
    #############################################@
    ####### MAP Restoration
    
    # MAP classification (Viterbi algo)
    X_MAP = getMAPClassif(K, N, Y, mu, sigma, I, t)
    
    # error rate compuation
    ConfMatrix_MAP, ERGlobal_MAP, ERbyClass_MAP = getConfMat(K, N, X, X_MAP)
    print('Confusion matrix for MAP:\n', ConfMatrix_MAP)
    print('Global error rate for MAP:', ERGlobal_MAP)
    print('By class error rate for MAP:', ERbyClass_MAP)
    