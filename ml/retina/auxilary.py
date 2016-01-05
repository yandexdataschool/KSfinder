from theano import tensor as T
import theano

__doc__ = """helper functions for symbolic theano code"""

_norm = lambda x: T.sqrt((x**2).sum(axis=-1,keepdims=True))
def _normalize(x):
    norms = _norm
    return T.switch(T.eq(norms,0),0,x/norms)

def _append_dim(_arg):
    return _arg.reshape([i for i in _arg.shape]+[1])
