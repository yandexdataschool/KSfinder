from sklearn.externals import joblib
import numpy as np
from sklearn.utils.extmath import cartesian
from theano import config as cfg

from retina_symbolic import *


__doc__ = "a bunch of theano functions compiled using retina_symboic.py expressions.  Deprecated since 1.04."


def retina_view(x0,y0,z0, alpha0, beta0, sigma=1, 
                grid_w = 100, grid_h = 100,
                dalpha = np.pi/3,dbeta = np.pi/3):
    
    alphas = np.linspace(alpha0-dalpha/2., alpha0+dalpha/2., grid_w,dtype=cfg.floatX)
    betas = np.linspace(beta0-dbeta/2., beta0+dbeta/2., grid_h,dtype=cfg.floatX)
    
    x0,y0,z0 = map(lambda v:np.array([v],dtype=cfg.floatX),[x0,y0,z0])
    
    grid = cartesian([x0,y0,z0,alphas,betas])
    

    return  [grid[:,i].astype(cfg.floatX) for i in range(grid.shape[1])]+[sigma]


def get_response_function(x0,y0,z0,alpha,beta,sigma = None,allow_downcast = True):
    '''compiles a function that computes retina response with
    fixed zero points (x0,y0,z0) and spheric angles(alpha,beta).
    The function compiled takes hits coordinates (x,y,z vectors) as input
    and returns activations for hidden units in respective order.
    
    If sigma parameter is set to None (default), the output function also takes
    retina sigma as an input. Otherwise, it is added to givens (sigma has to be a number)
    '''
    #initialize response parameters
    
    #grid
    _x0 = T.vector('retina.x0',"floatX")
    _y0 = T.vector('retina.y0',"floatX")
    _z0 = T.vector('retina.z0',"floatX")
    _alpha = T.vector('retina.alpha0',"floatX")
    _beta = T.vector('retina.beta0',"floatX")
    
    givens = {
        _x0:x0.astype(cfg.floatX),
        _y0:y0.astype(cfg.floatX),
        _z0:z0.astype(cfg.floatX),
        _alpha:alpha.astype(cfg.floatX),
        _beta:beta.astype(cfg.floatX),
    }
    
    #args
    _xarg = T.vector('point.x',"floatX")
    _yarg = T.vector('point.y',"floatX")
    _zarg = T.vector('point.z',"floatX")
    inputs = [_xarg,_yarg,_zarg]

    
    #sigma
    _sigma_var = T.scalar("retina.sigma","float32") #hard-code here

    if sigma is None:
        inputs.append(_sigma_var)
    else:
        givens[_sigma_var] = np.float32(sigma)
        
    #retina response
    _response = _compute_retina_response(_xarg,_yarg,_zarg,
                             _x0,_y0,_z0,
                             _alpha, _beta,
                            _sigma_var,
                     )
    
    return theano.function(inputs = inputs,
                              outputs=_response,
                              givens = givens,
                              allow_input_downcast=allow_downcast,
                             )

def get_response_with_shared_hits(_xhits=None,_yhits=None,_zhits=None):
    """returns a tuple of (compiled function, shared variables)
    where
       compiled function is a theano function of
                            _x0c,_y0c,_z0c, #scalar zero point coordinates
                            _alpha0,_dalpha, #scalar center and half-window along alpha 
                            _beta0,_dbeta, #scalar center and half-window along beta
                            _sigma, #scalar sigma (square root of variance)
                            _xdim,_ydim #scalar retina grid dimensions along alpha and beta
        using shared variables _xhits,_yhits,_zhits
        to represent event hits' coordinates
        
        shared variables are a list of variables [_xhits,_yhits,_zhits] (see later)
        
    If _xhits, _yhits or _zhits parameters are set to something other than None,
    they are treated as symbolic shared (or constant) variables for a function.
        
    If they are equal to None(default), new shared variables of type theano.config.floatX
    are created

    """
    #grid params
    _x0c = T.scalar('retina.x0.center',"floatX")
    _y0c = T.scalar('retina.y0.center',"floatX")
    _z0c = T.scalar('retina.z0.center',"floatX")
    _alpha0 = T.scalar('retina.alpha.center',"floatX")
    _beta0 = T.scalar('retina.beta.center',"floatX")
    _dalpha = T.scalar('retina.alpha.halfwindow',"floatX")
    _dbeta = T.scalar('retina.beta.halfwindow',"floatX")
    _sigma = T.scalar('retina.sigma',"floatX")

    _xdim = T.scalar("retina.dim.x","int32")
    _ydim = T.scalar("retina.dim.y","int32")


    ##create grid given parameters
    _n_points = _xdim*_ydim
    _x0 = T.repeat(_x0c,_n_points)
    _y0 = T.repeat(_y0c,_n_points)
    _z0 = T.repeat(_z0c,_n_points)

    #inner grid:
    #d0,d1,d2,d0,d1,d2,d0,d1,d2,d0,d1,d2
    _alpha = _linspace(_alpha0-_dalpha, _alpha0+_dalpha,_xdim)
    _alpha = _append_dim(_alpha)
    _alpha = T.repeat(_alpha,_ydim,axis=1).T.ravel()

    #outer grid:
    #d0,d0,d0,d0,d1,d1,d1,d1,d2,d2,d2,d2
    _beta = _linspace(_beta0-_dbeta, _beta0+_dbeta,_ydim)
    _beta = T.repeat( _beta,_xdim)


    ##event hits
    _shared = lambda name,val,dtype: theano.shared(val.astype(dtype),name,
                                               strict = True,allow_downcast=True)

    if _xhits is None:
        _xhits = _shared('hits.x',np.zeros(1),cfg.floatX)
    if _yhits is None:
        _yhits= _shared('hits.y',np.zeros(1),cfg.floatX)
    if _zhits is None:
        _zhits = _shared('hits.z',np.zeros(1),cfg.floatX)
    shareds = [_xhits,_yhits,_zhits]


    _response = _compute_retina_response(_xhits,_yhits,_zhits,
                         _x0,_y0,_z0,
                         _alpha, _beta,
                        _sigma,
                 )
    response = theano.function([
                            _x0c,_y0c,_z0c,
                            _alpha0,_dalpha,
                            _beta0,_dbeta,
                            _sigma,
                            _xdim,_ydim
                           ],_response)
    return response,shareds



def _response_job(retina,*args,**kwargs):
    return retina.response(*args,**kwargs)

class Retina:
    def __init__(self,ks,alphas,bethas,variance=0.01):
        """a fully vectorized retina for 3d lines going through a fixed point"""
        ks_x,ks_y,ks_z = np.vstack(ks)
        self.ks=ks
        grid = cartesian([ks_x,ks_y,ks_z,alphas,bethas])

        sigma = variance**.5 #because reasons

        self._respond = get_response_function(*grid.T,sigma = sigma)

        
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
            x,y,z=hits.T
            response = self._respond(x,y,z)
            return response
        
        

import os   
import pandas as pd

def retinize_events(event_names,event_folder,retina_views,
                    max_hits_block = 5000,max_hits_batch = 100000,report_rate=100):
    """computes several retina images for all events in event_prefices
    event_prefices - "runId_eventId"-like event name
    event_folder - a folder where all events are stored
    retina_views - an array of either lib.retina_compiled.retina_view or lib.retina_symbolic._retina_view 
                denoting retina grid parameters. Can be substituted with equivalent vector tuples for each retina:
                x,y,z,alpha,beta,sigma
    max_hits_block - a maximum amount of hits to be processed at one. This can prevent OutOfMemory errors if 
    computing on a low-ram GPU.
    max_hits_batch - how many hits (et maxima) are alowed for a batch of events evaluated in a single functon call.
                    Note that if a single event has more hits than max_hits_batch, it will still be processed in a single-event batch.
    report_rate - print progress each report_rate events. None means never report
    
    warning: this function compiles a theano graph with the given retinas thus you can't use it in further theano expressions.
    """
    #compiling function
    
    evtbatch = EventBatch(15000)
    
    retina_responses = [evtbatch.apply_retina(*r) for r in retina_views]
    
    process_event_batch = theano.function([],T.concatenate(retina_responses, axis=1))
    
    
    def get_hits(evt_id):
        return pd.DataFrame.from_csv(os.path.join(event_folder,evt_id+".hits.csv"))

    
    #processing events
    next_event_i = 0
    retina_images = []


    while True:

        batch_hits = []
        total_hits=0
        while True:
            evt_name = event_names[next_event_i]
            evt_hits = get_hits(evt_name)

            if len(batch_hits) != 0 and (len(evt_hits) + total_hits) > max_hits_batch:
                break
            else:
                batch_hits.append(evt_hits)
                next_event_i+=1
                total_hits += len(evt_hits)
                if next_event_i >= len(event_names):break
                if next_event_i % report_rate ==0:
                    print "processing events: %i/%i" % (next_event_i,len(event_names))

        batch_hit_xyz = [evt_df[["X","Y","Z"]].values.T for evt_df in batch_hits]


        evtbatch.load_events(batch_hit_xyz)
        retina_image_batch = process_event_batch()


        retina_images.append(retina_image_batch)

        if next_event_i >= len(event_names):break

    return np.vstack(retina_images)

