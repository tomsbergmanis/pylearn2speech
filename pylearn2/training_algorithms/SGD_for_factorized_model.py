from __future__ import division
import logging
import warnings
import numpy as np

from theano import config
from theano import function, function_dump
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values
from theano import tensor as T
from theano.tensor.nlinalg import MatrixInverse as MI
from pylearn2.monitor import Monitor
from pylearn2.space import CompositeSpace, NullSpace
from pylearn2.train_extensions import TrainExtension
from pylearn2.training_algorithms.training_algorithm import TrainingAlgorithm
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.utils.iteration import is_stochastic
from pylearn2.utils import py_integer_types, py_float_types
from pylearn2.utils import safe_zip
from pylearn2.utils import serial
from pylearn2.utils import sharedX
from pylearn2.utils.data_specs import DataSpecsMapping
from pylearn2.utils.timing import log_timing


log = logging.getLogger(__name__)

class SGD_for_factorized_model(SGD):
    def __init__(self, learning_rate, cost=None, batch_size=None,
                 monitoring_batches=None, monitoring_dataset=None,
                 monitor_iteration_mode='sequential',
                 termination_criterion=None, update_callbacks=None,
                 init_momentum=None, set_batch_size=False,
                 train_iteration_mode=None, batches_per_iter=None,
                 theano_function_mode='FAST_RUN', monitoring_costs=None,
                 seed=[2012, 10, 5]):
        """
            WRITEME

            learning_rate: The learning rate to use.
                            Train object callbacks can change the learning
                            rate after each epoch. SGD update_callbacks
                            can change it after each minibatch.
            cost: a pylearn2.costs.cost.Cost object specifying the objective
                  function to be minimized.
                  Optionally, may be None. In this case, SGD will call the model's
                  get_default_cost method to obtain the objective function.
            init_momentum: if None, does not use momentum
                            otherwise, use momentum and initialize the
                            momentum coefficient to init_momentum.
                            Callbacks can change this over time just like
                            the learning rate.

                            If the gradient is the same on every step, then
                            the update taken by the SGD algorithm is scaled
                            by a factor of 1/(1-momentum).

                            See section 9 of Geoffrey Hinton's "A Practical
                            Guide to Training Restricted Boltzmann Machines"
                            for details.
            set_batch_size: if True, and batch_size conflicts with
                            model.force_batch_size, will call
                            model.set_batch_size(batch_size) in an attempt
                            to change model.force_batch_size
            theano_function_mode: The theano mode to compile the updates function with.
                            Note that pylearn2 includes some wraplinker modes that are
                            not bundled with theano. See pylearn2.devtools. These
                            extra modes let you do things like check for NaNs at every
                            step, or record md5 digests of all computations performed
                            by the update function to help isolate problems with nondeterminism.

            Parameters are updated by the formula:

            inc := momentum * inc - learning_rate * d cost / d param
            param := param + inc
        """

        if isinstance(cost, (list, tuple, set)):
            raise TypeError("SGD no longer supports using collections of Costs to represent "
                            " a sum of Costs. Use pylearn2.costs.cost.SumOfCosts instead.")

        self.learning_rate = sharedX(learning_rate, 'learning_rate')
        self.cost = cost
        self.batch_size = batch_size
        self.set_batch_size = set_batch_size
        self.batches_per_iter = batches_per_iter
        self._set_monitoring_dataset(monitoring_dataset)
        self.monitoring_batches = monitoring_batches
        self.monitor_iteration_mode = monitor_iteration_mode
        if monitoring_dataset is None:
            if monitoring_batches is not None:
                raise ValueError("Specified an amount of monitoring batches but not a monitoring dataset.")
        self.termination_criterion = termination_criterion
        self.init_momentum = init_momentum
        if init_momentum is None:
            self.momentum = None
        else:
            assert init_momentum >= 0.
            assert init_momentum < 1.
            self.momentum = sharedX(init_momentum, 'momentum')
        self._register_update_callbacks(update_callbacks)
        if train_iteration_mode is None:
            train_iteration_mode = 'shuffled_sequential'
        self.train_iteration_mode = train_iteration_mode
        self.first = True
        self.rng = np.random.RandomState(seed)
        self.theano_function_mode = theano_function_mode
        self.monitoring_costs = monitoring_costs

    def train(self, dataset):
        if not hasattr(self, 'sgd_update'):
            raise Exception("train called without first calling setup")

        # Make sure none of the parameters have bad values
        for param in self.params:
            value = param.get_value(borrow=True)
            if np.any(np.isnan(value)):
                raise Exception("NaN in " + param.name)
            if np.any(np.isinf(value)):
                raise Exception("INF in " + param.name)

        self.first = False
        rng = self.rng
        if not is_stochastic(self.train_iteration_mode):
            rng = None

        data_specs = self.cost.get_data_specs(self.model)

        # The iterator should be built from flat data specs, so it returns
        # flat, non-redundent tuples of data.
        mapping = DataSpecsMapping(data_specs)
        space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
        source_tuple = mapping.flatten(data_specs[1], return_tuple=True)
        if len(space_tuple) == 0:
            # No data will be returned by the iterator, and it is impossible
            # to know the size of the actual batch.
            # It is not decided yet what the right thing to do should be.
            raise NotImplementedError("Unable to train with SGD, because "
                                      "the cost does not actually use data from the data set. "
                                      "data_specs: %s" % str(data_specs))
        flat_data_specs = (CompositeSpace(space_tuple), source_tuple)

        iterator = dataset.iterator(mode=self.train_iteration_mode,
                                    batch_size=self.batch_size,
                                    data_specs=flat_data_specs, return_tuple=True,
                                    rng=rng, num_batches=self.batches_per_iter)

        iter, freq = 0, 1000
        on_load_batch = self.on_load_batch

        for batch in iterator:

            for callback in on_load_batch:
                callback(mapping.nest(batch))
            self.sgd_update(*batch)
            # iterator might return a smaller batch if dataset size
            # isn't divisible by batch_size
            # Note: if data_specs[0] is a NullSpace, there is no way to know
            # how many examples would actually have been in the batch,
            # since it was empty, so actual_batch_size would be reported as 0.
            actual_batch_size = flat_data_specs[0].np_batch_size(batch)
            self.monitor.report_batch(actual_batch_size)
            for callback in self.update_callbacks:
                callback(self)

            iter += 1

            if iter % freq == 0:
                for param in self.params:
                    value = param.get_value(borrow=True)
                    if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                        raise Exception("Nan in " + param.name)

        for param in self.params:
            value = param.get_value(borrow=True)
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                raise Exception("Nan in " + param.name)

    def setup(self, model, dataset):

        if self.cost is None:
            self.cost = model.get_default_cost()

        inf_params = [param for param in model.get_params() if np.any(np.isinf(param.get_value()))]
        if len(inf_params) > 0:
            raise ValueError("These params are Inf: " + str(inf_params))
        if any([np.any(np.isnan(param.get_value())) for param in model.get_params()]):
            nan_params = [param for param in model.get_params() if np.any(np.isnan(param.get_value()))]
            raise ValueError("These params are NaN: " + str(nan_params))
        self.model = model

        batch_size = self.batch_size
        if hasattr(model, "force_batch_size"):
            if model.force_batch_size > 0:
                if batch_size is not None:
                    if batch_size != model.force_batch_size:
                        if self.set_batch_size:
                            model.set_batch_size(batch_size)
                        else:
                            raise ValueError(
                                "batch_size argument to SGD conflicts with model's force_batch_size attribute")
                else:
                    self.batch_size = model.force_batch_size
        model._test_batch_size = self.batch_size
        self.monitor = Monitor.get_monitor(model)
        self.monitor._sanity_check()

        data_specs = self.cost.get_data_specs(self.model)
        mapping = DataSpecsMapping(data_specs)
        space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
        source_tuple = mapping.flatten(data_specs[1], return_tuple=True)

        # Build a flat tuple of Theano Variables, one for each space.
        # We want that so that if the same space/source is specified
        # more than once in data_specs, only one Theano Variable
        # is generated for it, and the corresponding value is passed
        # only once to the compiled Theano function.
        theano_args = []
        for space, source in safe_zip(space_tuple, source_tuple):
            name = '%s[%s]' % (self.__class__.__name__, source)
            arg = space.make_theano_batch(name=name, batch_size=self.batch_size)
            theano_args.append(arg)
        theano_args = tuple(theano_args)

        # Methods of `self.cost` need args to be passed in a format compatible
        # with data_specs
        nested_args = mapping.nest(theano_args)
        fixed_var_descr = self.cost.get_fixed_var_descr(model, nested_args)
        self.on_load_batch = fixed_var_descr.on_load_batch

        cost_value = self.cost.expr(model, nested_args,
                                    **fixed_var_descr.fixed_vars)
        if isinstance(cost_value, tuple):
            cost_value, args = cost_value
            if isinstance(args, tuple):
                hp, cost_vector = args
            else:
                hp = args
        if cost_value is not None and cost_value.name is None:
            cost_value.name = 'objective'

        # Set up monitor to model the objective value, learning rate,
        # momentum (if applicable), and extra channels defined by
        # the cost
        learning_rate = self.learning_rate
        if self.monitoring_dataset is not None:
            self.monitor.setup(
                dataset=self.monitoring_dataset,
                cost=self.cost,
                batch_size=self.batch_size,
                num_batches=self.monitoring_batches,
                extra_costs=self.monitoring_costs,
                mode=self.monitor_iteration_mode
            )
            dataset_name = self.monitoring_dataset.keys()[0]
            monitoring_dataset = self.monitoring_dataset[dataset_name]
            self.monitor.add_channel(name='learning_rate',
                                     ipt=None,
                                     val=learning_rate,
                                     data_specs=(NullSpace(), ''),
                                     dataset=monitoring_dataset)
            if self.momentum:
                self.monitor.add_channel(name='momentum',
                                         ipt=None,
                                         val=self.momentum,
                                         data_specs=(NullSpace(), ''),
                                         dataset=monitoring_dataset)

        params = list(model.get_params())
        assert len(params) > 0
        for i, param in enumerate(params):
            if param.name is None:
                param.name = 'sgd_params[%d]' % i
        grads, updates = self.cost.get_gradients(model, nested_args,
                                                 **fixed_var_descr.fixed_vars)

        for param in grads:
            if grads[param].name is None and cost_value is not None:
                grads[param].name = ('grad(%(costname)s, %(paramname)s)' %
                                     {'costname': cost_value.name,
                                      'paramname': param.name})

        lr_scalers = model.get_lr_scalers()

        for key in lr_scalers:
            if key not in params:
                raise ValueError("Tried to scale the learning rate on " + \
                                 str(key) + " which is not an optimization parameter.")

        log.info('Parameter and initial learning rate summary:')
        for param in params:
            param_name = param.name
            if param_name is None:
                param_name = 'anon_param'
            lr = learning_rate.get_value() * lr_scalers.get(param, 1.)
            log.info('\t' + param_name + ': ' + str(lr))

        _, Y = nested_args
        for param in params:
            if param.name is not None:

                name =  param.name.split("_")[-1]
                if name == "U":
                    Up = param
                    lr = 2 * learning_rate

                    updates[Up] = Up - lr  * T.dot(T.dot(Up, hp.T), hp)
                    Up_new = updates[Up]

                elif name == "Ui":

                    Uip = param
                    lr = 2 * learning_rate / (1 -2 * learning_rate * T.mean(T.dot( hp,hp.T)))
                    term = T.dot(T.dot(Uip,hp.T), hp)
                    updates[Uip] = Uip + lr * term
                    Uip_new = updates[Uip]

                elif name == "V":
                    Vp = param
                    lr = 2 * learning_rate

                    updates[param] = Vp + lr * T.dot(Y.T,T.dot(Uip_new,hp.T).T)

                elif name == "Q":
                    Qp = param

                    h_hat = T.dot(Qp,hp.T)
                    y_hat =  T.dot(Up.T, T.dot(Vp.T,Y.T))
                    z_hat = h_hat - y_hat

                    lr1 = 2 * learning_rate
                    lr2 = lr1 * lr1 * cost_value

                    first_term = T.dot(hp.T, z_hat.T) + T.dot(z_hat, hp)
                    second_term = T.dot( hp.T,hp)
                    updates[Qp] = Qp - lr1 * first_term + lr2 * second_term.T

                else:
                    updates[param] = param - learning_rate * lr_scalers.get(param, 1.) * grads[param]

        for param in params:
            if param in updates and updates[param].name is None:
                updates[param].name = 'sgd_update(' + param.name + ')'


        model.censor_updates(updates)
        for param in params:
            update = updates[param]

            if update.name is None:
                update.name = 'censor(sgd_update(' + param.name + '))'
            for update_val in get_debug_values(update):
                if np.any(np.isinf(update_val)):
                    raise ValueError("debug value of %s contains infs" % update.name)
                if np.any(np.isnan(update_val)):
                    raise ValueError("debug value of %s contains nans" % update.name)


        with log_timing(log, 'Compiling sgd_update'):
            self.sgd_update = function(theano_args,
                                       updates=updates,
                                       name='sgd_update',
                                       on_unused_input='ignore',
                                       mode=self.theano_function_mode)


        self.params = params


class Stabiliser(object):

    def __init__(self, freqU=1000, freqQ=400  ):
        if freqQ > freqU:
            self._freqU = freqU
            self._freqQ = freqQ
        else:
            self._freqU = freqQ
            self._freqQ = freqQ
        self._mbcount = 0

    def __call__(self, algorithm):
        params_dict = {}
        if ((self._mbcount % self._freqU) == 0 or (self._mbcount % self._freqQ) == 0) and self._mbcount != 0 :
            params = algorithm.model.get_params()
            for param in params:
               if param.name is not None:
                   params_dict[param.name] = param.get_value()
                   name =  param.name.split("_")[-1]
                   if name == "U":
                       Up = param
                   elif name == "Ui":
                       Uip = param
                   elif name == "V":
                       Vp = param
                   elif name == "Q":
                       Qp = param
            if self._freqU != self._freqQ:
                if (self._mbcount % self._freqU) == 0:
                    W = np.dot(Vp.get_value(), Up.get_value())
                    params_dict[Vp.name] = W
                    d = Up.get_value().shape[0]
                    params_dict[Up.name] = np.identity(d, dtype="float32")
                    params_dict[Uip.name] = np.identity(d, dtype="float32")

                else:
                        W = np.dot(Vp.get_value(), Up.get_value())
                        params_dict[Qp.name] = np.dot(W.T, W)
            else:
                W = np.dot(Vp.get_value(), Up.get_value() )
                params_dict[Vp.name] = W
                #print "W",W
                d = Up.get_value().shape[0]
                params_dict[Up.name] = np.identity(d, dtype="float32")
                params_dict[Uip.name] = np.identity(d, dtype="float32")
                params_dict[Qp.name] = np.dot(W.T, W)
                #print  "U", Up.get_value()
                #print 	"Ui", Uip.get_value()

            algorithm.model.set_params(params_dict)
        self._mbcount += 1
