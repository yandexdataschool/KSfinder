#coding: utf-8 -*-
__doc__="""several more-or-less successfull layer designs used in the lasagne NN architecture"""


import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import MergeLayer,Gate
from lasagne import nonlinearities,init
from lasagne.utils import unroll_scan

floatX = theano.config.floatX
_shared = lambda name,val,dtype: theano.shared(val.astype(dtype),name,
                                               strict = True,allow_downcast=True)


class BaseEnvironment:
    """needs reworking... like seriously reworking"""
    def __init__(self): 
        """an environment class that defines what does MDP interact with"""
        raise NotImplementedError
    def decision_by_hidden(self,id,hidden_activation): #has to be deterministic function
        """take 1D float32 array and return action code.
        Action code must be stackable with hidden_activation
        Do not use random INSIDE. It's okay to pre-generate randomness"""
        raise NotImplementedError
    def input_by_decision(self,id,decision): #has to be deterministic function
        """take action code and return the next input observation.
        Do not use random INSIDE. It's okay to pre-generate randomness"""
        raise NotImplementedError
    def decision_init_factory(self,shape):
        """returns a value for '-1st' action, that is than aligned to the zero-tick
        hidden state"""
        return T.zeros(shape[0]+[2],dtype='floatX')
    def on_new_session(self,function):
        """takes a function of (batch_size, input_size) and
        calls it before each new session start"""
        raise NotImplementedError                 

class GRULayer(MergeLayer):
    r"""
    a recurrent layer that implements MDP network principles over base Gate Recurrent Unit architecture
    """
    def __init__(self, incoming, num_units,
                 
                 environment,
                 hidden_to_output_network = None,
                 resetgate=Gate(W_cell=None),
                 updategate=Gate(W_cell=None),
                 hidden_update=Gate(W_cell=None,
                                    nonlinearity=nonlinearities.tanh),
                 
                 
                 
                 hid_init_factory=init.Constant(0.),
                 
                 grad_clipping=5.,
                 only_return_final=False,
                 **kwargs):

        incomings = [incoming]

        
        # Initialize parent layer
        super(GRULayer, self).__init__(incomings, **kwargs)

        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final

        
        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        #functions
        self.env = environment
        self.env.on_new_session(lambda batch_size,seq_len: 
                                self.reset_state(batch_size,seq_len))

        if self.name is None:
            self.name = ""
        name = self.name

        #shared variables
        input_shape = [ (i if i is not None else 1) for i in input_shape]
        
        
        batch_size,_,_,_ = input_shape
        
        
        self.hid_init_factory = hid_init_factory
        hidden_zero_batch = self.hid_init_factory((batch_size,num_units))
        
        self.hid_init_batch = _shared(name+".hid_init_shared_batch",
                                hidden_zero_batch,floatX)

        
        decision_zero = self.env.decision_init_factory()
        decision_zero_batch = np.repeat(decision_zero.reshape(1,-1),
                                        batch_size,axis=0)
        

        self.decision_init_batch = _shared(name+".decision_init_shared_batch",
                                decision_zero_batch,decision_zero.dtype)
        #/shared variables
        
        
        #hidden_to_output_network: register and copy params
        self.hidden_to_output_network = hidden_to_output_network
        if self.hidden_to_output_network is not None:
            _ihnn_layers = lasagne.layers.get_all_layers(self.hidden_to_output_network)
            _ihnn_params_dict = {par:tags for layer in _ihnn_layers for (par,tags) in layer.params.items()}

            for param,tags in _ihnn_params_dict.items():
                self.add_param(param,param.shape.eval(),param.name)
        ###
        
        
        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in all parameters from gates
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')
        
    def reset_state(self,batch_size,seq_length):
        """causes re-initialization of shared variables that exist within session scope
        [like "last state", "last decision",etc.]"""
                
        hidden_zero_batch = self.hid_init_factory((batch_size,self.num_units))
        self.hid_init_batch.set_value(hidden_zero_batch)
        
        
        decision_zero = self.env.decision_init_factory()
        decision_zero_batch = np.repeat(decision_zero.reshape(1,-1),
                                        batch_size,axis=0)

        self.decision_init_batch.set_value(decision_zero_batch)






    def get_output_shape_for(self, input_shapes):
        #pattern: batch_i,sequence_i,channel_i,unit_i
        #channels are [decision codes,activations] correspondingly
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], 2,self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1],2, self.num_units

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable
        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable. 
            
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        print "called get_output"
        assert len(inputs) ==1
        
        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)
            
            

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape



        
        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)


        
        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        
        # Create single recurrent computation step function
        # hid_previous was the previous state of hidden/output layer.
        # decision_previous is self.env.decision_by_hidden(hid_previous)
        
        def step(step_iter,hid_previous,output_previous,decision_previous, *args):
            
            input_data = self.env.input_by_decision(step_iter,decision_previous)
            
            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping:
                input_data = theano.gradient.grad_clip(
                    input_data, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
            input_n = T.dot(input_data, W_in_stacked) + b_stacked

            # Reset and update gates
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)
            hidden_update = hidden_update_in + resetgate*hidden_update_hid
            
            if self.grad_clipping:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update)

            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate)*hid_previous + updategate*hidden_update
            
            #apply hid to output network if it exists, else just sent hidden layer as output
            if self.hidden_to_output_network is None:
                output = hid
            else:
                output = self.hidden_to_output_network.get_output_for(hid)
            
            decision = self.env.decision_by_hidden(step_iter,output)
            
            return hid,output,decision


        #iterator, required for Qlearning stuff
        sequences = [T.arange(seq_len)]
        
        # The hidden-to-hidden weight matrix is always used in step
        # As we don't precompute inputs, we have to
        # provide the input weights and biases to the step function
        non_seqs = [W_hid_stacked,W_in_stacked, b_stacked] 
        non_seqs += self.env.dependencies
        non_seqs += lasagne.layers.get_all_params(self.hidden_to_output_network)
        
        #output initializers
        if self.hidden_to_output_network is None:
            response_init = self.hid_init_batch
        else:
            response_init = self.hidden_to_output_network.get_output_for(self.hid_init_batch)
        outputs_info = [self.hid_init_batch,
                        response_init,
                        self.decision_init_batch]

        
        # Scan op iterates over first dimension of input and repeatedly
        # applies the step function
        (hid_out,resp_out,dec_out) = theano.scan(
            fn=step,
            sequences=sequences,#replaced by n_steps
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            strict=True,
        )[0]
        
        
        
        hid_out = hid_out.reshape([seq_len,num_batch,-1])
        resp_out = resp_out.reshape([seq_len,num_batch,-1])
        dec_out = dec_out.reshape([seq_len,num_batch,-1])
        

                
        
        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            return hid_out[-1],resp_out[-1],dec_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)
            resp_out = resp_out.dimshuffle(1, 0, 2)
            dec_out = dec_out.dimshuffle(1, 0, 2)

            return hid_out,resp_out,dec_out

def _compute_qvalues_naive(_rewards,_is_alive,_gamma_or_gammas,dependencies=[],strict = True):
    """ computes Qvalues assuming all actions are optimal
    params:
        _rewards: immediate rewards floatx[batch_size,time]
        _is_alive: whether the session is still active int/bool[batch_size,time]
        _gamma_or_gammas: delayed reward discount number, scalar or vector[batch_size]
        dependencies: everything you need to evaluate first 3 parameters (only if strict==True)
        strict: whether to evaluate Qvalues using strict theano scan or non-strict one
    returns:
        Qvalues: floatx[batch_size,time]
            the regular Qvalue is computed via:
                Qs(t,action_at(t) ) = _rewards[t] + _gamma_or_gammas* Qs(t+1, best_action_at(t+1))
            using out assumption on optimality:
                best_action_at(t) = action(t)
            this essentially becomes:
                Qs(t,picked_action) = _rewards[t] + _gamma_or_gammas* Qs(t+1,next_picked_action)
            and does not require actions themselves to compute (given rewards)
            
    """
    outputs_info = [T.zeros_like(_rewards[:,0]),]
    non_seqs = [_gamma_or_gammas]+dependencies

    sequences = [_rewards.T,_is_alive.T] #transpose to iterate over time, not over batch

    def backward_qvalue_step(rewards,isAlive, next_Qs,*args):
        this_Qs = T.switch(isAlive,
                               rewards + _gamma_or_gammas * next_Qs, #assumes optimal next action
                               0.
                          )
        return this_Qs

    _reference_Qvalues = theano.scan(backward_qvalue_step,
                sequences=sequences,
                non_sequences=non_seqs,
                outputs_info=outputs_info,

                go_backwards=True,
                strict = strict
               )[0] #shape: [time_seq_inverted,batch]

    return _reference_Qvalues.T[:,::-1] #[batch,time_seq]

def get_reference_tuples(_rewards,_isalive,_decision_ids,_activations,
                               _gamma_or_gammas = _shared('q_learning_gamma',np.float32(0.99),floatX),
                               end_code=None,
                               naive = False
                        ):
    """computes three vectors:
      action IDs (1d,single integer) vector of all actions that were commited during all sequences
      in the batch, concatenated in one vector... that is, excluding the ones that happened
      after the end_code action was sent.
    
      Qpredicted - activations for action_IDs ONLY at each time 
      before sequence end action was committed for each sentence in the batch (concatenated) 
      but for the last time-slot.
      
      Qreference - sum over immediate rewards and gamma*predicted activations for
      next round after first vector predictions
      if naive == True, all actions are considered optimal in terms of Qvalue """
    _alive_selector = _isalive[:,:-1].nonzero() # number of: [batch,time] but for (seq_len)-th



    # predictions for Qvalues in the order of _alive_selector (id, n_outputs)
    _predicted_Qvalues = _activations[_alive_selector]
    
    #iterator array over all prediction samples
    _event_i = T.arange(_predicted_Qvalues.shape[0])
    # corresponding choices
    _chosen_action_IDs = _decision_ids[_alive_selector]
    
    #predictions that resulted in actions
    _tested_predicted_Qvalues = _predicted_Qvalues[_event_i, :]#_chosen_action_IDs]

    # corresponding rewards (immediate)
    _immediate_rewards = _rewards[_alive_selector]
    
    
    if not naive:
        _next_selector = (_alive_selector[0],_alive_selector[1]+1) # number of: [batch,time+1]

        # corresponding rewards for all next turn actions
        _predicted_next_rewards =_activations[_next_selector]

        #best of _predicted_next_rewards for each time

        _optimal_next_rewards = T.max(_predicted_next_rewards,axis=1)
        #if session end is given, zero out future rewards after session end
        if end_code is not None:
            _optimal_next_rewards = T.switch( T.eq(_chosen_action_IDs,end_code),
                                                     0.,
                                                     _optimal_next_rewards
                                             )
        #full rewards from taking _chosen_action and behaving optimally later on
        _reference_Qvalues = _immediate_rewards + _gamma_or_gammas*_optimal_next_rewards
    else:#naive
        _reference_Qvalues = _compute_qvalues_naive(_rewards,_isalive,_gamma_or_gammas)[_alive_selector]
    return _chosen_action_IDs,_immediate_rewards, _tested_predicted_Qvalues,_reference_Qvalues