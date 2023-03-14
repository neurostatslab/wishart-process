# -*- coding: utf-8 -*-
"""
@author: Amin
"""

from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import numpy as np
import scipy as sp

from scipy import stats

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

# %%
def draw_ellipse(mean,covariance,colors,std_devs=3,ax=None,line_width=2):
    '''sample grid that covers the range of points'''
    min_p = mean - std_devs*np.sqrt(np.diag(covariance))
    max_p = mean + std_devs*np.sqrt(np.diag(covariance))
    
    x = np.linspace(min_p[0],max_p[0],256) 
    y = np.linspace(min_p[1],max_p[1],256)
    X,Y = np.meshgrid(x,y)
    
    Z = multivariate_normal.pdf(
        np.stack((X.reshape(-1),Y.reshape(-1))).T, 
        mean=mean, cov=(std_devs**2)*covariance, allow_singular=True)
    Z = Z.reshape([len(x),len(y)])
    
    if ax is None:
        plt.contour(X, Y, Z, 0,colors=colors,linewidth=line_width)
    else:
        ax.contour(X, Y, Z, 0,colors=colors,linewidths=line_width)

# %%
def visualize_pc(
        means,covs,pc=None,title_str='',fontsize=9,dotsize=30,
        std=3,lim=None,save=False,file=None
    ):
    '''Visualize point clouds and atlases learned from them
    '''

    if means.shape[2] > 2:
        pca = PCA(n_components=2)
        pca.fit_transform(np.vstack(means))

        if pc is not None:
            pc = pca.transform(pc)
        
        means = [pca.transform(means[i]) for i in range(len(means))]
        covs = [pca.components_@covs[i]@pca.components_.T for i in range(len(means))]
            
            

    fig = plt.figure(figsize=(15,8))
    plt.title(title_str)

    colors = np.vstack((
        np.zeros((2,len(means))),
        np.linspace(0,1,len(means))
    )).T
    
    plt.plot(
        np.mean(means,1)[:,0],np.mean(means,1)[:,1]
    )


    for j in range(len(means)):
        plt.scatter(
            means[j][:,0],means[j][:,1], 
            s=10, c=colors[j], marker='.'
        )
        
        draw_ellipse(
            means[j].mean(0)[:2],covs[j][:2,:2],colors[j],
            std_devs=std,line_width=1
        )

    if pc is not None:
        plt.scatter(pc[:,0],pc[:,1])

    
    # plt.axis('equal')

    if lim is not None:
        plt.xlim(lim)
        plt.ylim(lim)
    
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()

# %%
def plot_loss(loss,error=None,xlabel='',ylabel='',titlestr='',colors=None,legends=None,fontsize=15,linewidth=2,save=False,file=None):
    if colors is None: colors = plt.cm.hsv(np.linspace(0,1,len(legends)+1)[0:-1])[:,0:3]

    plt.figure(figsize=(10,3))
    
    for i in range(len(loss)):
        plt.plot(loss[i],color=colors[i],linewidth=linewidth)
        plt.yscale('log') 
        if error is not None:
            plt.fill_between(np.arange(len(loss[i])), loss[i]-error[i], loss[i]+error[i], color=colors[i], alpha=.1)

    plt.grid('on')
    
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    
    if legends is not None:
        plt.legend(legends,fontsize=fontsize)
    plt.title(titlestr,fontsize=fontsize)
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()


# %%
def plot_box(performance,fontsize=10,save=False,file=None):

    plt.boxplot(performance.values())
    plt.xticks(np.arange(1,1+len(performance)),list(performance.keys()),fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()