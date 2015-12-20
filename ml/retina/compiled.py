from theano import tensor as T
import theano
__doc__="A fully vectorized theanized solution to massively computing line-point distances and points from arrays of lines"

def normalize(x):
    norms = T.sqrt((x**2).sum(axis=-1,keepdims=True))
    return T.switch(T.eq(norms,0),0,x/norms)

floatx="float32"
def align_arg(_arg,length):
    v = T.repeat(_arg.reshape([-1,1]),length,axis =1)
    return v
def append_dim(_arg):
    return _arg.reshape([i for i in _arg.shape]+[1])


#grid
_x0 = T.vector('x0',floatx)
_y0 = T.vector('y0',floatx)
_z0 = T.vector('z0',floatx)
_alpha = T.vector('alpha',floatx)
_beta = T.vector('beta',floatx)

_power = T.scalar("power",floatx)
_variance = T.scalar("sigma",floatx)
#args
_xarg = T.vector('x',floatx)
_yarg = T.vector('y',floatx)
_zarg = T.vector('z',floatx)
_x,_y,_z =  map(lambda _v: align_arg(_v,_x0.shape[0]),[_xarg,_yarg,_zarg])


##preprocess
_anchor_point = T.horizontal_stack(*map(append_dim,[_x0,_y0,_z0]))

_sin_alpha = T.sin(_alpha)
_sin_beta = T.sin(_beta)
_cos_alpha = T.cos(_alpha)
_cos_beta = T.cos(_beta)

_dx = _sin_alpha*_cos_beta
_dy = _sin_beta*_cos_alpha
_dz = T.ones_like(_dx)


_dirvec = T.horizontal_stack(
    append_dim(_dx),
    append_dim(_dy),
    append_dim(_dz))

#assert False
_dirvec = normalize(_dirvec).reshape([1,_dirvec.shape[0],_dirvec.shape[1]])

##call
_x_of_z = (_x0 + (_z-_z0)*_dx)
_y_of_z = (_y0 + (_z-_z0)*_dy)


_call_from_z = T.concatenate([
        append_dim(_x_of_z),
        append_dim(_y_of_z),
        append_dim(_z)],axis=-1)
##distance
_v = T.concatenate(map(append_dim,[_x,_y,_z]),axis=-1) #pt_id,line_id,xyz

_anchor_to_p_vec = normalize(_v-_anchor_point)

_costheta = T.sum(_dirvec *_anchor_to_p_vec,axis=-1)

_acos = T.arccos(_costheta)
_acos = T.switch(T.isnan(_acos),0,_acos)

_vec_norms = T.sqrt((
        (_v-_anchor_point)**2
    ).sum(axis=-1,keepdims=False))

_dist = _vec_norms*T.sin(_acos)
_dist = T.switch(T.eq(_power%2,0),_dist,T.abs_(_dist))

##retina_response
##TODO:  switch to save computations for points with high dist
#min_influence = 0.01
#max_distance = min_influence**(1./power) ...

_response = T.sum( 
    T.exp(-_dist**_power /_variance),
    axis=0)


_f_onetime_response = theano.function(inputs = [_xarg,_yarg,_zarg,_x0,_y0,_z0,_alpha,_beta,_power,_variance],
                                      outputs=_response,
                                      givens = [],
                                      allow_input_downcast=True,
)

def get_response_function(x0,y0,z0,alpha,beta):
    return theano.function(inputs = [_xarg,_yarg,_zarg,_power,_variance],
                              outputs=_response,
                              givens = {
                                        _x0:x0.astype(floatx),
                                        _y0:y0.astype(floatx),
                                        _z0:z0.astype(floatx),
                                        _alpha:alpha.astype(floatx),
                                        _beta:beta.astype(floatx),
                                        },
                           
                              allow_input_downcast=True,
                             )