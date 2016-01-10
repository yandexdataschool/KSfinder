import scipy
import scipy.stats
import numpy as np
__doc__ = '''a class that generates target vectors for neural network 
as a retina-like representation of inputs (KS decay coordinates) in a percentile space'''

from scipy.special import expit
ramp = lambda x,xc,w,slope=1.: (expit(x-xc+w/2.)*(1.-expit(x-xc-w/2.)))**slope


def Multiramp(width,slope=1.):
    
    def output_function(x,centers):
        x = x[...,None]
        centers = centers[None,...]
        stumps = ramp(x,centers,width,slope)
    
        return 1.- (1. - stumps).prod(axis=0)
    
    return output_function
def Gaussian(sigma):
    
    return   lambda x,centers: np.exp(
                            -(x[...,None]-centers[None,...])**2/sigma**2
                             ).sum(axis=0)


class PercentileSpaceMapper1D:
    def __init__(self,value_samples,n_centers=64,
                 smoothing = 0.5,offset = 0.1,
                 activation=None):
        '''a class that generates target vectors for neural network 
        as a retina-like representation of inputs (KS decay coordinates) in a percentile space'''
    
        value_samples = np.array(value_samples)

        self.n_fake_points = int(smoothing*len(value_samples))        
        self.bounds = np.percentile(value_samples,[0,100])
        fake_points = np.linspace(*self.bounds, num=self.n_fake_points)

        self.value_sample = np.concatenate([value_samples,fake_points])

        self.center_percentiles = np.linspace(0+offset,100-offset,n_centers)
        self.centers =  np.percentile(self.value_sample,self.center_percentiles )
        
        if activation is None:
            activation = Gaussian(5)
        self.activation = activation
    def get_activity_uniform(self,x):
        '''
        old: get activity based on euclidian distance
        single point restoration: 
            compute mean over center locations, weighted with their activity
        '''
        
        return self.activation(x,self.centers)

    def get_activity_percentile(self,x):
        '''
        get activity based on distance in percentile space
        single point restoration: 
            compute mean over center percentiles, weighted with their activity
            get the corresponding percentile of the sample points (including fake ones)
        '''

        val_to_percentile = lambda x:scipy.stats.percentileofscore(self.value_sample,x,kind= 'mean')

        x_percentiles = np.array(map(val_to_percentile,x) )

        return self.activation(x_percentiles,self.center_percentiles)

    