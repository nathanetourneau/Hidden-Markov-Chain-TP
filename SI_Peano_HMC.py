import numpy as np
from func import *

from PIL.Image import *

from Peano.PeanoImage import Peano
from Peano.InvPeanoImage import PeanoInverse

if __name__ == '__main__':
    # Set the parameters and read the data
    nbIter = 30

    # names of the sources and results images
    imagefilename = 'dday_128'
    imagename = f'./Peano/sources/{imagefilename}.png'
    resultname = f'./Peano/sources/{imagefilename}_reconstruction_{nbIter}_iter.png'
    curvename = f'./Peano/results/{imagefilename}_reconstruction_{nbIter}_iter'

    # Load image
    image = np.array(open(imagename), dtype=float)
    Y = Peano(image)

    # The data (simulated)
    [L, C] = image.shape
    N = L*C
    K = 7

    # Parameters of MM: mean, variance and a priori proba
    meanTabIter = np.zeros(shape=(nbIter, K))
    sigmaTabIter = np.zeros(shape=(nbIter, K))
    cTabIter = np.zeros(shape=(nbIter, K, K))
    tTabIter = np.zeros(shape=(nbIter, K, K))
    ITabIter = np.zeros(shape=(nbIter, K))

    # Error rate according to EM iterations
    ConfusionMatrixTab = np.zeros(shape=(nbIter, K, K))
    MeanErrorRateTab = np.zeros(shape=(nbIter))
    MeanErrorRateTabbyClass = np.zeros(shape=(nbIter, K))

    ##########################################################################
    # Parameters initialization
    iteration = 0
    print('--->iteration=', iteration)
    meanTabIter[iteration, :], sigmaTabIter[iteration,
                                            :], cTabIter[iteration, :, :] = InitParam(K, N, Y)
    tTabIter[iteration, :, :], ITabIter[iteration,
                                        :] = getProbaMarkov(K, cTabIter[iteration, :, :])

    # Proba computations
    alpha, S = getAlpha(K, N, Y, meanTabIter[iteration, :], sigmaTabIter[iteration,
                                                                         :], ITabIter[iteration, :], tTabIter[iteration, :, :])
    beta = getBeta(K, N, Y, meanTabIter[iteration, :], sigmaTabIter[iteration,
                                                                    :], ITabIter[iteration, :], tTabIter[iteration, :, :], S)
    gamma = getGamma(K, N, alpha, beta)

    # MPM classification
    X_MPM = getMPMClassif(N, gamma)

    ##########################################################################
    # EM iterations
    for iteration in range(1, nbIter):
        print('--->iteration=', iteration)

        gamma = EM_Iter(iteration, K, N, Y, meanTabIter,
                        sigmaTabIter, cTabIter, tTabIter, ITabIter)

        # MPM classification
        X_MPM = 255 / (K-1) * getMPMClassif(N, gamma)

    # Save generated signals
    image = PeanoInverse(X_MPM)

    reconstruction = fromarray(np.uint8(image))
    reconstruction.save(resultname)
