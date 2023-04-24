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

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

# %%
def visualize_pc(
        means,covs,pc=None,title_str='',fontsize=9,dotsize=30,
        std=3,lim=None,save=False,file=None
    ):
    '''Visualize point clouds and atlases learned from them
    '''

    if means.shape[2] > 2:
        if pc is not None:
            pca = PCA(n_components=2)
            pca.fit_transform(pc)
            pc = pca.transform(pc)
        else:
            pca = PCA(n_components=2)
            pca.fit_transform(np.vstack(means))
            
        
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
            means[j].mean(0)[:2],covs[j][:2,:2],colors[j],ax=plt.gca(),
            std_devs=std,line_width=1
        )

    if pc is not None:
        plt.scatter(pc[:,0],pc[:,1],s=.5)

    
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
def plot_box(performance,titlestr='',fontsize=10,save=False,file=None):

    plt.boxplot(performance.values())
    plt.xticks(np.arange(1,1+len(performance)),list(performance.keys()),fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(titlestr,fontsize=fontsize)
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()

# %%

def plot_tuning(x,y,lw=2,titlestr='',fontsize=10,save=False,file=None):
    plt.plot(x,y,lw=lw)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.title(titlestr,fontsize=fontsize)
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()



# %%
def draw_ellipse(mu, cov, colors, ax, std_devs=3.0, facecolor='none', **kwargs):
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, edgecolor=colors)

    scale_x = np.sqrt(cov[0, 0]) * std_devs
    scale_y = np.sqrt(cov[1, 1]) * std_devs

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mu[0], mu[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# %%
def visualize_covariances(cov,titlestr='',fontsize=10,save=False,file=None):
    fig, axes = plt.subplots(
        nrows=cov.shape[0], ncols=cov.shape[1],
        figsize=(3*cov.shape[1], 3*cov.shape[0])
    )
    if cov.shape[0] == 1: axes = axes[None]
    if cov.shape[1] == 1: axes = axes[:,None]
    
    vmin = cov.min()
    vmax = cov.max()
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            im = axes[i,j].imshow(cov[i,j], vmin=vmin, vmax=vmax)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.9, 0.1, 0.05, 0.8])
    fig.colorbar(im, cax=cbar_ax)

    plt.suptitle(titlestr,fontsize=fontsize)


    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()


# %%
def plot_torus(phi,theta,R0=2,a=.3,save=False,file=None):
    R0, a = 2., .3
    zlim = max(R0,a)+1

    # torus parametrization
    x_ = (R0 + a*np.cos(theta)) * np.cos(phi)
    y_ = (R0 + a*np.cos(theta)) * np.sin(phi)
    z_ = a * np.sin(theta)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_,y_,z_,s=10)
    ax.set_zlim(-zlim,zlim)

    ax.axis('off')
    ax.view_init(elev=30, azim=20)

    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()

