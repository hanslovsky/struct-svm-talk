#!/usr/bin/python

import numpy as np
import sklearn.svm
import matplotlib.pyplot as plt

def generateData( N, means, covs, seed ):
    data = np.empty( (0, len(means[0]) ) )
    labels = np.empty( (0,) )
    np.random.seed(seed)
    for idx, (mean, cov) in enumerate( zip(means, covs) ):
        data   = np.append( data, np.random.multivariate_normal( mean, cov, N ), axis = 0 )
        labels = np.append( labels, np.zeros(N) + idx, axis = 0 )
    return data, labels

def fitToData( data, labels, C ):
    classifier = sklearn.svm.SVC(kernel='linear', C=C)
    classifier.fit(data, labels)
    return classifier

def plotData( data, labels, classifier, mesh_step, with_margin, **kwargs ):
    if not data.shape[1] == 2:
        raise Exception("only plot 2 dimensional data!")
    
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy       = np.meshgrid(np.arange(x_min, x_max, mesh_step ),
                               np.arange(y_min, y_max, mesh_step ))

    Z = classifier.predict( np.c_[xx.ravel(), yy.ravel()] )
    Z = Z.reshape( xx.shape )

    x = np.linspace( x_min, x_max )
    w = classifier.coef_[0]
    a = -w[0] / w[1]
    y = a * x - (classifier.intercept_[0]) / w[1]

    yy_down = -1.0 / w[1] + a * x - classifier.intercept_[0] / w[1]
    yy_up   = 1.0 / w[1] + a * x - classifier.intercept_[0] / w[1]
    
    contour     = plt.contourf( xx, yy, Z, alpha=0.2, **kwargs )
    scatter     = plt.scatter(data[:, 0], data[:, 1], c = labels, **kwargs )
    if with_margin:
        margin      = plt.plot( x, y, 'k-' )
        margin_up   = plt.plot( x, yy_up, 'k--' )
        margin_down = plt.plot( x, yy_down, 'k--' )
        support     = plt.scatter(classifier.support_vectors_[:, 0],
                                  classifier.support_vectors_[:, 1],
                                  facecolors='none',
                                  s = 80)
    plt.axis( [x_min, x_max, y_min, y_max] )
    plt.axis('off')

    if with_margin:
        return scatter, contour, support, margin, margin_up, margin_down

    else:
        return scatter, contour
    

if __name__ == "__main__":
    data, labels = generateData(600, ([1., 0.], [4., 3.]), ([[0.8, 0],[0,0.6]],[[1.5, 0],[0,1]]), 100)
    C = 1.0
    classifier = fitToData( data, labels, C )
    plotData( data, labels, classifier, mesh_step = 0.02, cmap = 'cool', with_margin = True )
    plt.show()


    data, labels = generateData(600, ([1., 0.], [5.5, 0.5], [4., 3.]), ([[0.8, 0],[0,0.6]], [[2., 1], [0., 1.]], [[1.5, 0],[0,1]]), 100)
    C = 1.0
    classifier = fitToData( data, labels, C)
    plotData( data, labels, classifier, mesh_step = 0.02, cmap = 'cool', with_margin = False )
    plt.show()
