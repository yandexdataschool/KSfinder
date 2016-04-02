from theano import tensor as T
import theano
import numpy as np
__doc__="A fully vectorized theanized solution to massively computing line-point distances and points from arrays of lines"

from auxilary import _normalize, _append_dim, _norm,_linspace

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
    
    floatX = theano.config.floatX
    givens = {
        _x0:x0.astype(floatX),
        _y0:y0.astype(floatX),
        _z0:z0.astype(floatX),
        _alpha:alpha.astype(floatX),
        _beta:beta.astype(floatX),
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
    floatX = theano.config.floatX

    if _xhits is None:
        _xhits = _shared('hits.x',np.zeros(1),floatX)
    if _yhits is None:
        _yhits= _shared('hits.y',np.zeros(1),floatX)
    if _zhits is None:
        _zhits = _shared('hits.z',np.zeros(1),floatX)
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
