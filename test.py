import numpy as np
from func import *

from Peano.PeanoImage import Peano
from Peano.InvPeanoImage import PeanoInverse


if __name__ == '__main__':

    # @
    # Set the parameters and read the data
    nbIter = 30

    # names of the sources and results images
    imagename = './Peano/sources/cible_64_bruit.png'
    resultname = f'./Peano/sources/cible_64_reconstruction_{nbIter}_iter.png'
    curvename = f'./Peano/results/cible_64_reconstruction_{nbIter}_iter/'

    # Load image
    image = np.array(open(imagename), dtype=float)
    Y = Peano(image)

    # The data (simulated)
    N = Y.shape[0]
    K = 2
