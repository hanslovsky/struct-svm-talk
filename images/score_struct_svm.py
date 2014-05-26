#!/usr/bin/python2

import numpy as np
import matplotlib.pyplot as plt

class Gaussian(object):
    def __init__(self, mean, cov):
        m = np.array(mean)
        c = np.array(cov)
        assert m.shape == ( 2, )
        assert c.shape == ( 2, 2 )
        self.mean = m[:, np.newaxis]
        self.cov  = c
        self.norm = 1.0 / ( np.sqrt( np.linalg.det( self.cov ) ) * 2 * np.pi )

    def __call__(self, x):
        assert x.shape[0] == 2
        diff = x - self.mean
        return self.norm * np.exp( -0.5 * np.sum( diff * self.cov.dot( diff ), axis = 0 ) )

def plotObjective(gaussians, coefficients, border, mesh_step, **kwargs):
    def sumGaussians(gaussians, coefficients, x):
        return sum( c*g(x) for c,g in zip( coefficients, gaussians ) )

    def getMaxMinAxis(gaussians, border):
        maxAxis = np.max( [ g.mean for g in gaussians ], axis = 0 ).reshape((2,)) + border
        minAxis = np.min( [ g.mean for g in gaussians ], axis = 0 ).reshape((2,)) - border
        return minAxis[0], minAxis[1], maxAxis[0], maxAxis[1]

    x_min, y_min, x_max, y_max = getMaxMinAxis(gaussians, border)
    xx, yy       = np.meshgrid(np.arange(x_min, x_max, mesh_step ),
                               np.arange(y_min, y_max, mesh_step ))
    Z = sumGaussians( gaussians, coefficients, np.c_[xx.ravel(), yy.ravel()].transpose() )
    Z = Z / np.max( Z )
    Z = Z.reshape( xx.shape )


    contour = plt.contourf( xx, yy, Z, **kwargs )
    return contour, np.min(Z), np.max(Z)


g1 = Gaussian([5, 5], [[1, 0], [0, 1]])
g2 = Gaussian([3, 7], [[1, 0], [0, 1]])
g3 = Gaussian([4, 6], [[1, 0], [0., 1]])
g4 = Gaussian([2, 7.5], [[1, 0], [0, 1]])
# t  = np.arange(10).reshape((2, 5))
contour, minV, maxV = plotObjective([g1, g2, g3, g4], [1.0, -0.7,1.0, 1.0], border = 1.0, mesh_step = 0.005, alpha = 0.5, cmap = 'cool' )
# cb = plt.colorbar(contour)
# cb.set_ticks([], True)
plt.axis('off')
plt.show()



contour, minV, maxV = plotObjective([g1, g2, g3, g4], [1.0, -0.0,0.0, 0.0], border = 1.0, mesh_step = 0.005, alpha = 0.5, cmap = 'cool' )
# cb = plt.colorbar(contour)
# cb.set_ticks([], True)
plt.axis('off')
plt.show()

