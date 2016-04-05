__doc__ = """a humble theano function that computes maxima and minima over decay space prediction.
Used mainly for evaluation purposes"""

import numpy as np
import theano
import theano.tensor as T
import lasagne


_activity = T.tensor4('activity_for_maxima_computation',dtype='floatX')

_diff = T.extra_ops.diff(_activity,axis=-1)

_is_increasing = _diff>0
_extremum_sign = T.extra_ops.diff(_is_increasing,axis=-1)

_is_minima = _extremum_sign>0
_is_maxima = _extremum_sign<0

get_minima = theano.function([_activity],_is_minima.nonzero(),on_unused_input='ignore')
get_maxima = theano.function([_activity],_is_maxima.nonzero(),on_unused_input='ignore')
