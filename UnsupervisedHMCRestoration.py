import numpy as np
import matplotlib.pyplot as plt
import os
from func import *


if __name__ == '__main__':
    
    #############################################@
    # Set the parameters and read the data
    nbIter = 30

    # names of the sources and results images
    filenameorig = 'sources/XY_t4.out'
    basename = os.path.basename(filenameorig)
    filename, file_extension = os.path.splitext(basename)

    # The data (simulated)
    X, Y = np.loadtxt(filenameorig)
    N = X.shape[0]
    K = len(np.unique(X))
    
    # Parameters of MM: mean, variance and a priori proba
    meanTabIter  = np.zeros(shape=(nbIter, K))
    sigmaTabIter = np.zeros(shape=(nbIter, K))
    cTabIter     = np.zeros(shape=(nbIter, K, K))
    tTabIter     = np.zeros(shape=(nbIter, K, K))
    ITabIter     = np.zeros(shape=(nbIter, K))
    
    # Error rate according to EM iterations
    ConfusionMatrixTab      = np.zeros(shape=(nbIter, K, K))
    MeanErrorRateTab        = np.zeros(shape=(nbIter))
    MeanErrorRateTabbyClass = np.zeros(shape=(nbIter, K))

    ##########################################################################
    # Parameters initialization
    iteration = 0
    print('--->iteration=', iteration)
    meanTabIter[iteration, :], sigmaTabIter[iteration, :], cTabIter[iteration, :, :] = InitParam(K, N, Y)
    tTabIter[iteration, :, :], ITabIter[iteration, :] = getProbaMarkov(K, cTabIter[iteration, :, :])

    # Proba computations
    alpha, S = getAlpha(K, N, Y, meanTabIter[iteration, :], sigmaTabIter[iteration, :], ITabIter[iteration, :], tTabIter[iteration, :, :])
    beta     = getBeta(K, N, Y, meanTabIter[iteration, :], sigmaTabIter[iteration, :], ITabIter[iteration, :], tTabIter[iteration, :, :], S)
    gamma    = getGamma(K, N, alpha, beta)

    # MPM classification
    X_MPM = getMPMClassif(N, gamma)

    # error rate computation
    ConfusionMatrixTab[iteration, :, :], MeanErrorRateTab[iteration], MeanErrorRateTabbyClass[iteration] = getConfMat(K, N, X, X_MPM)

    print('Initial estimations:')
    print('  Confusion matrix for MPM =\n', ConfusionMatrixTab [iteration, :])
    print('  Global Error rate for MPM: ',  MeanErrorRateTab[iteration])
    print('  Class Error rate for MPM: ',   MeanErrorRateTabbyClass[iteration])

#    print('Mean estimated', meanTabIter[iteration, :])
#    print('Std estimated', sigmaTabIter[iteration, :])
#    print('c estimated', cTabIter[iteration, :])
#    print('t estimated', tTabIter[iteration, :])
#    print('I estimated', ITabIter[iteration, :])


    ##########################################################################
    # EM iterations
    for iteration in range(1, nbIter):
        print('--->iteration=', iteration)
        
        gamma = EM_Iter(iteration, K, N, Y, meanTabIter, sigmaTabIter, cTabIter, tTabIter, ITabIter)
        
        # MPM classification
        X_MPM = getMPMClassif(N, gamma)
        
        # error rate computation
        ConfusionMatrixTab[iteration, :, :], MeanErrorRateTab[iteration], MeanErrorRateTabbyClass[iteration] = getConfMat(K, N, X, X_MPM)
        # print('ITERATION ', iteration, ' over ', nbIter)
        # print('  Confusion matrix for MPM =\n', ConfusionMatrixTab [nbIter-1, :])
        # print('  Global Error rate for MPM: ', MeanErrorRateTab[nbIter-1])
        # print('  Class Error rate for MPM: ', MeanErrorRateTabbyClass[nbIter-1])
        # input('pause')
        
    # Drawing curves: evolution of parameters. Don't forget likelihood
    pathToSave = './results/'+filename
    print('pathToSave=', pathToSave)
    DrawCurvesParam(nbIter, K, pathToSave, meanTabIter, sigmaTabIter, tTabIter)
    DrawCurvesError(nbIter, K, pathToSave, MeanErrorRateTabbyClass, MeanErrorRateTab)
    
    print('Final estimations:')
    print('  Confusion matrix for MPM =\n', ConfusionMatrixTab [nbIter-1, :])
    print('  Global Error rate for MPM: ', MeanErrorRateTab[nbIter-1])
    print('  Class Error rate for MPM: ', MeanErrorRateTabbyClass[nbIter-1])
    
    # Save generated signals
    np.savetxt(pathToSave+'_EM_MPM.out', X_MPM)

#    print('Mean estimated', meanTabIter[nbIter-1, :])
#    print('Std estimated', sigmaTabIter[nbIter-1, :])
#    print('c estimated', cTabIter[nbIter-1, :])
#    print('t estimated', tTabIter[nbIter-1, :])
#    print('I estimated', ITabIter[nbIter-1, :])
    
    