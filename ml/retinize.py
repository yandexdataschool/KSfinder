
import os
import pandas as pd
from sklearn.externals import joblib
import random
from retina.retina_theanized import Retina
import numpy as np

__doc__="""
A tool that allows to get Retina representation of several events on a multicore machine. 
This approach is faster, than it would take to process each event successively using the same number of cores.
"""

def get_retina_response(hits_several,retina):
    """joblib thread method"""
    responses = []
    for hits in hits_several:
        response = retina.response(hits,block_size=None,
                    n_jobs=1,inner_block_size=None)
        responses.append(response)
    return responses

def retinize(hits_several,retina,n_closest = 3000,n_shards=32):
    """apply retina to a list of hit matrices"""
    nearest_hits_several = []
    for hits in hits_several:
        
        hits_dist = np.linalg.norm(hits-retina.ks,axis=-1)
        
        hits = hits[np.argsort(hits_dist)[:n_closest],:]
        nearest_hits_several.append(hits)
    
    responses=[]
    shard_size = (len(hits_several)-1)/n_shards +1
    
    for shard_i in range(n_shards):
        responses.append(joblib.delayed(get_retina_response)(
                nearest_hits_several[shard_i*shard_size:(shard_i+1)*shard_size],
                retina))
    
        
    responses = [resp for batch in joblib.Parallel(n_jobs = -1)(responses)
                             for resp in batch]
    retina_pts = np.vstack(responses)
    return retina_pts







def get_retina_response_by_filepaths(fpaths,retina,n_closest=5000):
    """joblib thread method"""
    hits_several=[]
    
    for i in range(len(fpaths)):
        fpath = fpaths[i]

        hits = pd.DataFrame.from_csv(fpath)[["X","Y","Z"]].values
        
        if len(hits)==0:
            continue
            
        hit_dists = np.linalg.norm(hits-retina.ks,axis=-1)
        
        hits = hits[np.argsort(hit_dists)[:n_closest],:]

        hits_several.append(hits)
    
    
    responses = []
    for hits in hits_several:
        response = retina.response(hits,block_size=None,
                    n_jobs=1,inner_block_size=None)
        responses.append(response)
    return responses



def retinize_folder(folder,retina,
                   out_file = None,
                   max_rows=float('inf'),
                   n_closest=5000,
                   n_shards = 32,
                   return_names = False):
    """apply retina files in a given folder"""
    
    fnames = os.listdir(folder)
    if max_rows < len(fnames):
        fnames = fnames[:max_rows]

    fpaths = map(lambda line:os.path.join(folder,line) , fnames)
        
    
    print "applying retina..."
    responses=[]
    shard_size = (len(fpaths)-1)/n_shards +1
    
    for shard_i in range(n_shards):
        responses.append(
            joblib.delayed(get_retina_response_by_filepaths)(
                fpaths[shard_i*shard_size:(shard_i+1)*shard_size],
                retina,
                n_closest = n_closest
            )
        )
    
        
    responses = [resp for batch in joblib.Parallel(n_jobs = -1)(responses)
                             for resp in batch]
    retina_pts = np.vstack(responses)
    if out_file is not None:
        print 'saving...'
        np.save(out_file,retina_pts)
        
    print 'done'
    return retina_pts if not return_names else (fnames,retina_pts)