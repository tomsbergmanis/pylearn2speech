__author__ = 's1044253'
import math
import sys
import warnings
import logging

import numpy as np
from theano import config
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.gradient import consider_constant
import theano.tensor as T
from theano import function
from theano.tensor.nlinalg import svd as SVD

from pylearn2.costs.mlp import Default
from pylearn2.expr.probabilistic_max_pooling import max_pool_channels
from pylearn2.linear import conv2d, conv1d
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
from pylearn2.expr.nnet import pseudoinverse_softmax_numpy
from pylearn2.space import CompositeSpace
from pylearn2.space import Conv2DSpace
from pylearn2.space import Space
from pylearn2.space import VectorSpace
from pylearn2.utils import function
from pylearn2.utils import py_integer_types
from pylearn2.utils import safe_union
from pylearn2.utils import safe_zip
from pylearn2.utils import sharedX
from pylearn2.models.mlp import Softmax

class Factorized(Softmax):
    def __init__(self,
                 n_classes,
                 layer_name,
                 irange = None,
                 b_lr_scale = None,
                 V_lr_scale = None,
                 U_lr_scale = None,
                 Q_lr_scale = None,
                 Ui_lr_scale = None
                 ):

        self.__dict__.update(locals())
        del self.self
        assert isinstance(n_classes, py_integer_types)

        self.output_space = VectorSpace(n_classes)

    def set_input_space(self, space):
        self.input_space = space

        if not isinstance(space, Space):
            raise TypeError("Expected Space, got "+
                    str(space)+" of type "+str(type(space)))

        self.input_dim = space.get_total_dimension()
        self.needs_reformat = not isinstance(space, VectorSpace)

        desired_dim = self.input_dim
        self.desired_space = VectorSpace(desired_dim)

        if not self.needs_reformat:
            assert self.desired_space == self.input_space

        rng = self.mlp.rng

        self._params = []
        V = np.zeros((self.n_classes, self.input_dim),dtype=np.float32)
        self.V = sharedX(V,   self.layer_name + "_V" )

        U = np.identity( self.input_dim)
        self.U = sharedX(U, self.layer_name + "_U")

        Q =  np.zeros((self.input_dim, self.input_dim),dtype=np.float32)
        self.Q = sharedX(Q, self.layer_name + "_Q")

        Ui =  np.identity(self.input_dim,dtype=np.float32)
        self.Ui = sharedX(Ui, self.layer_name + "_Ui")

        self._params = [ self.U, self.Ui, self.V, self.Q]

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.needs_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)

        for value in get_debug_values(state_below):
            if self.mlp.batch_size is not None and value.shape[0] != self.mlp.batch_size:
                raise ValueError("state_below should have batch size "+str(self.dbm.batch_size)+" but has "+str(value.shape[0]))

        self.desired_space.validate(state_below)
        assert state_below.ndim == 2

        W = T.dot(self.V, self.U)
        assert W.ndim == 2

        Z = T.dot(state_below, W.T)

        rval = Z

        for value in get_debug_values(rval):
            if self.mlp.batch_size is not None:
                assert value.shape[0] == self.mlp.batch_size

        return (rval, state_below)

    def get_params(self):
        rval = []
        rval.append(self.U)
        rval.append(self.Ui)
        rval.append(self.V)
        rval.append(self.Q)

        return rval

    def get_lr_scalers(self):

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        if not hasattr(self, 'V_lr_scale'):
            self.V_lr_scale = None

        if not hasattr(self, 'U_lr_scale'):
            self.U_lr_scale = None

        if not hasattr(self, 'Q_lr_scale'):
            self.Q_lr_scale = None

        if not hasattr(self, 'Ui_lr_scale'):
            self.Ui_lr_scale = None

        rval = OrderedDict()

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        if self.V_lr_scale is not None:
            rval[self.V] = self.V_lr_scale

        if self.U_lr_scale is not None:
            rval[self.U] = self.U_lr_scale

        if self.Q_lr_scale is not None:
            rval[self.Q] = self.Q_lr_scale

        if self.Ui_lr_scale is not None:
            rval[self.Ui] = self.Ui_lr_scale

        return rval

    def cost(self, Y, Y_hat):
        Y_hat_true, h = Y_hat
        assert hasattr(Y_hat_true, 'owner')
        owner = Y_hat_true.owner
        assert owner is not None
        val = SqLoss()([h, self.Q, self.U, self.Ui, self.V, Y])[0]
        return (T.mean(val,  dtype='float32'), (h, T.mean(val, axis=0)))

    def get_monitoring_channels(self):
        W = T.dot(self.V,self.U)
        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([
                            ('row_norms_min'  , row_norms.min()),
                            ('row_norms_mean' , row_norms.mean()),
                            ('row_norms_max'  , row_norms.max()),
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                            ])

    def censor_updates(self, updates):
        pass

class SqLoss(T.Op):
    __props__ = ()

    def make_node(self, inputs):
        assert hasattr(self, '_props'), "Your version of theano is too old to support __props__."
        SqLoss.view_map = {1: [1],
            2: [2],
            3: [3],
            4: [4],
            5: [0]}
        return T.Apply(self, inputs, [i.type() for i in inputs])

    def perform(self, node, inputs, output_storage):
        h, Q, U, Ui, V, y = inputs
        z = output_storage
        h_hat = np.dot(Q, h.T)
        y_hat = np.dot(U.T, np.dot(V.T, y.T))
        cost =  np.dot(h, h_hat -2 * y_hat) + np.dot(y, y.T)
        #print "cost", cost
        z[0][0] = cost
        z[1][0] = Q
        z[2][0] = U
        z[3][0] = Ui
        z[4][0] = V
        z[5][0] = h

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        h, Q, U, Ui, V, y = inputs

        h_hat = T.dot(Q, h.T)
        y_hat =  T.dot(U.T, T.dot(V.T, y.T))
        dh =  2 * (h_hat - y_hat)
        return [dh.T, Q, U, Ui, V, h]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
