from theano import tensor as T
import theano
import numpy as np
__doc__="A fully vectorized theanized solution to massively computing line-point distances and points from arrays of lines"

from auxilary import _normalize, _append_dim, _norm,_linspace


cfg = theano.config

def _align_arg(_arg,length):
    """retina-specific dimension reshape that duplicates arg along new appended axis"""
    v = T.repeat(_arg.reshape([-1,1]),length,axis =-1)
    return v


def _stack(*args):
    """stacks argument symbolic vectors along last axis
    after appending fake axes to each of them
    """
    return T.concatenate(map(_append_dim,args),axis = -1)




def _get_direction_components(_alpha,_beta):
    """symbolic expression for directional vectors of all lines by their spheric angles,
    !ALARM! the coordinate system used here is a bit different from canonical.
    See angle axes for details
    alpha - "horizontal"(azimuth) spheric angle within XZ (!) plane
    beta - "vertical"(zenite) spheric angle within YZ (!) plane
    returns: expressions for vector components(_dx,_dy,_dz) as a tuple.
    
    The output is guaranteed contain length-one (unit) vectors only.
    """

    _sin_alpha = T.sin(_alpha)
    _sin_beta = T.sin(_beta)
    _cos_alpha = T.cos(_alpha)
    _cos_beta = T.cos(_beta)


    _dx = _sin_alpha
    _dy = _cos_alpha*_sin_beta
    _dz = _cos_alpha * _cos_beta
    
    return _dx,_dy,_dz




##call
def _get_point_by_z(_z,
                    _x0,_y0,_z0,
                    _dx,_dy,_dz,):
    """symbolic expression for coordinates of points on retina line grid lines 
    as a function of Z and a retina grid,
    _z - target Z for points
    _x0,_y0,_z0 - base points' coordinates
    _dx,_dy,_dz - direction vector components
    direction vector components are obtainable via
    _dx,_dy,_dz = _get_direction_components(_alpha,_beta)"""
    _x_of_z = (_x0 + (_z-_z0)*_dx/_dz)
    _y_of_z = (_y0 + (_z-_z0)*_dy/_dz)


    _coords_by_z = _stack(_x_of_z,_y_of_z,_z,axis=-1)
    return _coords_by_z






##distance
def _compute_distance_squared(_x,_y,_z,
                      _x0,_y0,_z0,
                      _alpha, _beta,
                     ):
    """
    symbolic expression for distances between each point and each line
    _x,_y,_z - vectors of hits' coordinates
    _x0,_y0,_z0 - retina base points' coordinates
    _dx,_dy,_dz - direction vector components
    direction vector components are obtainable via
    _dx,_dy,_dz = _get_direction_components(_alpha,_beta)"""

    #align hits to match
    num_lines = _x0.shape[0]
    
    _x,_y,_z =  map(lambda _coord: _align_arg(_coord,num_lines),[_x,_y,_z])
    _v = _stack(_x,_y,_z) #indices: pt_id,line_id,xyz
    
    
    
    #get direction vectors
    _dx,_dy,_dz = _get_direction_components(_alpha,_beta)
    
    _dirvec = _stack(_dx,_dy,_dz)

    #get vectors from retina zero point (p0) to each hit
    
    _anchor_point = _stack(_x0,_y0,_z0)
    _anchor_to_p_vec = _v-_anchor_point
    
    #get the length of projection of _anchor_to_p_vec on directional vector
    _projection = T.sum(_dirvec *_anchor_to_p_vec,axis=-1,keepdims = True)

    #scale directional vector by that much
    _dirvec_scaled = _dirvec * _projection

    #get distance by substracting vectors
    _dist_vec = _anchor_to_p_vec - _dirvec_scaled
    _dist_squared = (_dist_vec*_dist_vec).sum(axis=-1,keepdims=False)
    return _dist_squared

def _compute_retina_response(_x,_y,_z,
                             _x0,_y0,_z0,
                             _alpha, _beta,
                            _sigma,
                     ):
    """
    symbolic expression for retina response
    _x,_y,_z - vectors of hits' coordinates
    _x0,_y0,_z0 - retina base points' coordinates
    _dx,_dy,_dz - direction vector components
    direction vector components are obtainable via
    _dx,_dy,_dz = _get_direction_components(_alpha,_beta)
    sigma - retina variance (not yet squared)
    """
    #distances
    _dist_squared = _compute_distance_squared(_x,_y,_z,
                      _x0,_y0,_z0,
                      _alpha, _beta,
                     )
    
    ##retina_response

    _variance = _sigma**2
    _response = T.sum( 
        T.exp(-_dist_squared /_variance),
        axis=0)
    return _response


def _retina_view(x0c,y0c,z0c,alpha0,beta0,sigma=1,
                grid_w=100,grid_h=100,
                dalpha= np.pi/3, dbeta= np.pi/3,
                dtype=cfg.floatX):
    ##create grid given parameters
    n_points = grid_w*grid_h
    
    
    check_format = lambda v:  T.constant(v,dtype=dtype) if not hasattr(v,'dtype') else v
    
    x0c,y0c,z0c,alpha0,beta0,sigma = map(check_format,[x0c,y0c,z0c,alpha0,beta0,sigma])
    
    x0 = T.repeat(x0c,n_points)
    y0 = T.repeat(y0c,n_points)
    z0 = T.repeat(z0c,n_points)
    alpha0 = alpha0.astype(dtype)
    beta0 = beta0.astype(dtype)

    #inner grid:
    #d0,d1,d2,d0,d1,d2,d0,d1,d2,d0,d1,d2
    alpha = _linspace(alpha0-dalpha,alpha0+dalpha,grid_w)
    alpha = _append_dim(alpha)
    alpha = T.repeat(alpha,grid_w,axis=1).T.ravel()

    #outer grid:
    #d0,d0,d0,d0,d1,d1,d1,d1,d2,d2,d2,d2
    beta = _linspace(beta0-dbeta, beta0+dbeta,grid_h)
    beta = T.repeat(beta,grid_h)
    
    
    return [x0,y0,z0,alpha,beta,sigma]

    

    
#EventBatch
class EventBatch:
    """a class that wraps shared variables for event hits. 
    Used to compute retina output for all of them.
    Everything it says is symbolic"""
    def __init__(self,max_block_size = 10**20):
        """max_block_size- maximum amount of hits stacked in a single block. Reduce to save gpu memory (15k ~ 8gb max)"""
        
        #batch borders : ids of first elements of each batch concatenated with [n_hits]
        # 0,5232, 17231, ..., len(event_x)
        self.block_borders = theano.shared(np.zeros(2,dtype='int32'),
                                      name= "blocks") 
        self.block_to_event = theano.shared(np.zeros(1,dtype='int32'),
                                      name= "block_to_event") 

        self.event_x = theano.shared(np.zeros(5,dtype=cfg.floatX),
                                      name= "event_x") 
        self.event_y = theano.shared(np.zeros(5,dtype=cfg.floatX),
                                      name= "event_y") 
        self.event_z = theano.shared(np.zeros(5,dtype=cfg.floatX),
                                      name= "event_z") 
        
        self.max_block_size = max_block_size
        
    def load_events(self,event_xyz_list):
        
        x_cat = []
        y_cat = []
        z_cat = []
        
        max_block_size = self.max_block_size
        
        block_borders = [0]
        block_to_event=[]
        
        for i, (x,y,z) in enumerate(event_xyz_list):
            
            while len(x) > max_block_size:
                x_cat.append(x[:max_block_size])
                y_cat.append(y[:max_block_size])
                z_cat.append(z[:max_block_size])
                
                block_borders.append(block_borders[-1]+max_block_size)
                block_to_event.append(i)
                
                x = x[max_block_size:]
                y = y[max_block_size:]
                z = z[max_block_size:]
                
                
            x_cat.append(x)
            y_cat.append(y)
            z_cat.append(z)
            block_borders.append(block_borders[-1]+ len(x))
            block_to_event.append(i)
            
        
                    
        self.event_x.set_value(np.concatenate(x_cat).astype(cfg.floatX))
        self.event_y.set_value(np.concatenate(y_cat).astype(cfg.floatX))
        self.event_z.set_value(np.concatenate(z_cat).astype(cfg.floatX))
        self.block_borders.set_value(np.array(block_borders,dtype='int32'))
        self.block_to_event.set_value(np.array(block_to_event,dtype='int32'))
        
            
    def apply_retina(self,x0,y0,z0, alpha,beta,sigma):
        
        
        def retina_for_block(batch_start, batch_end,evt_x,evt_y,evt_z,
                            x0,y0,z0,alpha,beta,sigma):
            
            x,y,z = evt_x[batch_start:batch_end],evt_y[batch_start:batch_end],evt_z[batch_start:batch_end]

            return _compute_retina_response(x,y,z,x0,y0,z0,alpha,beta,sigma)

        sequences = [self.block_borders[:-1], self.block_borders[1:]]
        
        non_sequences = [self.event_x,self.event_y,self.event_z,
                         x0,y0,z0,alpha,beta,sigma]
        
        block_responses = theano.map(retina_for_block,sequences=sequences,non_sequences=non_sequences)[0]
        
        n_events = self.block_to_event[-1]+1
        
        def add_blocks_to_event(evt_i, block_to_evt, block_responses):
            
            sel = T.eq(block_to_evt,evt_i).nonzero()
            return T.sum( block_responses[sel],axis=0)
        
        responses = theano.map(add_blocks_to_event,
                               sequences=[T.arange(n_events)],
                               non_sequences=[self.block_to_event,
                                              block_responses])[0]

        return responses
                         

