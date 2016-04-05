from sklearn.externals import joblib
import numpy as np
from line_grid import LineGrid

from sklearn.utils.extmath import cartesian

def _response_job(retina,*args,**kwargs):
    return retina.response(*args,**kwargs)

class Retina:
    def __init__(self,ks,alphas,bethas,variance=0.01,power=2,):
        """a fully vectorized retina for 3d lines going through a fixed point"""
        ks_x,ks_y,ks_z = np.vstack(ks)
        self.ks=ks
        grid = cartesian([ks_x,ks_y,ks_z,alphas,bethas])
        
        self.linegrid = LineGrid(*grid.T)
        self.variance = variance
        self.power = power
        
    def response(self,hits,block_size = None ,n_jobs = 1,inner_block_size = None):
        """compute a retina response matrix [alpha,beta] -> response at (alpha,beta)"""
        
        if block_size is not None and len(hits) > block_size:
            n_blocks = (len(hits) -1)/ block_size+1


            block_responses = []
            for i in range(n_blocks):
                hit_block = hits[block_size*i:block_size*(i+1)]
                if n_jobs ==1:
                    block_response = self.response(hit_block,
                                                     block_size=None,n_jobs=1)
                else:
                    block_response = joblib.delayed(_response_job)(self,hit_block,
                                                     block_size=inner_block_size,n_jobs=1)
                block_responses.append(block_response)

            if n_jobs != 1:
                block_responses = joblib.Parallel(n_jobs = n_jobs)(block_responses)

            response = np.sum(block_responses,axis = 0)
            return response

        else: #single block

            dists = self.linegrid.distance_from(*hits.T)
            if self.power%2!=0:
                dists = np.abs(dists)
            response = np.sum(
                            np.exp(-dists**self.power /self.variance),
                                axis=0)
            return response