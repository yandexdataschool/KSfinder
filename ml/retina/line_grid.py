import numpy as np
from collections import Iterable
normalize = lambda x: x/np.linalg.norm(x,axis=-1,keepdims=True)

__doc__="A fully vectorized solution to massively computing line-point distances and points from arrays of lines"

class LineGrid:
    def __init__(self,x0,y0,z0,alpha,beta):
        """i implement a set of lines in the retinous(spherical) coordinate space."""
        
        
        self.n_lines = 1
        x0,y0,z0,alpha,beta = map(self.check_allignment,
                                  [x0,y0,z0,alpha,beta])

        x0,y0,z0,alpha,beta = map(self.adjust_allignment,
                                  [x0,y0,z0,alpha,beta])
        
        self.x0,self.y0,self.z0 = x0,y0,z0
        
        self.anchor_point = np.vstack([x0,y0,z0]).T    
        
        assert len(alpha) == len(beta) == self.anchor_point.shape[0]
        
        
        sin_alpha = np.sin(alpha)
        sin_beta = np.sin(beta)
        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)

        self.dx = sin_alpha*cos_beta
        self.dy = sin_beta*cos_alpha
        self.dz = np.ones(self.dx.shape)
        
        self.dirvec = normalize(np.array([self.dx,self.dy,self.dz]).T)[np.newaxis,...]
        
    def check_allignment(self,par):
        if isinstance(par,Iterable):
            if self.n_lines ==1 or len(par) == self.n_lines:
                self.n_lines = len(par)
            else:
                raise ValueError, "parameters misalligned"
        else: par = [par]
        return par
    def adjust_allignment(self,par):
        if len(par) == 1:
            return np.array(par)[np.newaxis,...]
        elif len(par) == self.n_lines:
            return np.array(par)
        else:
            raise ValueError,"internal error when adjusting param allignments"
    def __call__(self,z):
        """returns tensor[z_id,line_id,"xyz"] -> xyz coordinates with z = z_id for line_id"""
        z = np.vstack( [z for i in self.dx]).T
        return np.concatenate([
            (self.x0 + (z-self.z0)*self.dx)[...,np.newaxis],
            (self.y0 + (z-self.z0)*self.dy)[...,np.newaxis],
            z[...,np.newaxis],
            ],axis =  -1)
    def distance_from(self,x,y,z):
        """returns matrix[pt_id,line_id] -> euclidian dist between pt_id'th point and line_id'th line"""
        x = np.vstack( [x for i in self.dx]).T[...,np.newaxis]
        y = np.vstack( [y for i in self.dx]).T[...,np.newaxis]
        z = np.vstack( [z for i in self.dx]).T[...,np.newaxis]
                
        v = np.concatenate([x,y,z],axis=-1) #pt_id,line_id,xyz
        
        anchor_to_p_vec = normalize(v-self.anchor_point)
        anchor_to_p_vec[np.isnan(anchor_to_p_vec)] = 0 #in case point is at anchor point
        
        costheta = np.sum(self.dirvec *anchor_to_p_vec,axis=-1)
        
        acos = np.arccos(costheta)
        acos[np.isnan(acos)] = 0
        
        dist = np.linalg.norm(v-self.anchor_point,axis=-1)*np.sin(acos)
        return dist
