import numpy as np
from PIL.Image import *
import matplotlib.pyplot as plt

try:
    from hilbertcurve import HilbertCurve
except ModuleNotFoundError:
    from Peano.hilbertcurve import HilbertCurve


def getPowerOfTwo(n):
    if (n == 0):
        return False
    cpt = 0
    while (n != 1):
        if (n % 2 != 0):
            return -1
        n = n // 2
        cpt += 1

    return cpt


def Peano(image):

    # lecture de l'image
    [L, C] = np.shape(image)
    N = L*C
    p = getPowerOfTwo(L)
    # print('N=', N, ', p=', p)

    hilbert_curve = HilbertCurve(p, 2)
    vecteur = np.zeros(shape=(N))
    for ii in range(N):
        l, c = hilbert_curve.coordinates_from_distance(ii)
        vecteur[ii] = image[l, c]

    return vecteur


if __name__ == '__main__':
    imagename = './sources/cible_64_bruit.png'
    image = np.array(open(imagename), dtype=float)
    vecteur = Peano(image)
    np.savetxt('./sources/image.out', vecteur)
