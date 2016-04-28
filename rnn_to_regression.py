#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: rnn_to_regression.py
#Author: yuxuan
#Created Time: 2016-04-24 19:00:28
############################
from __future__ import print_function
import sys
import time
import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict
import six.moves.cPickle as pickle

SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

def _p(pp, name):
    return '%s_%s' % (pp, name)

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)

def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    #Wi,Wf,Wo,Wc四个为了并行计算
    W = numpy.concatenate([ortho_weight(options['dim_input']),
                           ortho_weight(options['dim_input']),
                           ortho_weight(options['dim_input']),
                           ortho_weight(options['dim_input'])], axis=1)
    params[_p(prefix, 'W')] = W
    #Ui,Uf,Uo,Uc四个为了并行计算
    U = numpy.concatenate([ortho_weight(options['dim_input']),
                           ortho_weight(options['dim_input']),
                           ortho_weight(options['dim_input']),
                           ortho_weight(options['dim_input'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_input'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params

def init_params(options):
    """
    除LSTM外的所有参数
    """
    params = OrderedDict()
    params = param_init_lstm(options, params)
    #regession
    params['U'] = 0.01 * numpy.random.randn(options['dim_input'],
            options['dim_output']).astype(config.floatX)
    params['b'] = numpy.zeros((options['dim_output'],)).astype(config.floatX)
    #输入升维
    params['U_input'] = 0.01 * numpy.random.randn(options['dim_inout'],
            options['dim_input']).astype(config.floatX)
    params['b_input'] = numpy.zeros((options['dim_input'],)).astype(config.floatX)

    return params

def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

#def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
def lstm_layer(tparams, state_below, options, prefix='lstm'):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    #assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    #def _step(m_, x_, h_, c_):
    def _step(x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_input']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_input']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_input']))
        c = tensor.tanh(_slice(preact, 3, options['dim_input']))

        c = f * c_ + i * c
        #c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        #h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_input = options['dim_input']
    rval, updates = theano.scan(_step,
                                #sequences=[mask, state_below],
                                sequences=[state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_input),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_input)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]

def adadelta(lr, tparams, grads, x, y, cost):
#def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tparams: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]
    '''
    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')
    '''
    f_grad_shared = theano.function([x, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

#def rmsprop(lr, tparams, grads, x, mask, y, cost):
def rmsprop(lr, tparams, grads, x, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    '''
    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')
    '''
    f_grad_shared = theano.function([x, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update

def build_model(tparams, options):
    trng = RandomStreams(SEED)
    x = tensor.matrix('x', dtype=config.floatX)
    #y = tensor.vector('y', dtype='int64')
    y = tensor.matrix('y', dtype=config.floatX)

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]
    #add this word

    x = x.reshape([x.shape[0], x.shape[1], options['dim_inout']])
    tempx = tensor.dot(x, tparams['U_input']) + tparams['b_input']
    #x = x.reshape([x.shape[0], x.shape[1], options['dim_input']])

    #proj = lstm_layer(tparams, x, options, prefix='lstm', mask=mask)
    proj = lstm_layer(tparams, tempx, options, prefix='lstm')
    
    #mean pooling
    #proj = (proj * mask[:, :, None]).sum(axis=0)
    proj = proj.sum(axis=0)
    #proj = proj / mask.sum(axis=0)[:, None]
    
    pred = tensor.dot(proj, tparams['U']) + tparams['b']
    #theano.printing.debugprint(pred)
    #theano.printing.pydotprint(pred, outfile="pics/logreg_pydotprint_pred.png", var_with_name_simple=True)
    #f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred_prob = theano.function([x], pred, name='f_pred_prob')
    cost = tensor.mean((y - pred) ** 2)
    #return x, mask, y, f_pred_prob, cost
    return x, y, f_pred_prob, cost

def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0
    #mask = None
    for _, valid_index in iterator:
        x, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index])
        #pred_probs = f_pred_prob(x, mask)
        pred_probs = f_pred_prob(x)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs

def pred_error(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    #mask = None
    for _, valid_index in iterator:
        x, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index])
        #preds = f_pred_prob(x, mask)
        preds = f_pred_prob(x)
        targets = numpy.array(data[1])[valid_index]
        valid_err += ((targets - preds)**2).sum()  
    valid_err = numpy_floatX(valid_err) / len(data[0])

    return valid_err

def load_data(test_size=0.1, valid_portion=0.1):
    import pandas as pd
    from random import random

    def _load_data(data, n_prev = 100):
        """
        data should be pd.DataFrame()
        """
        
        docX, docY = [], []
        for i in range(len(data)-n_prev):
            docX.append(data.iloc[i:i+n_prev].as_matrix())
            docY.append(data.iloc[i+n_prev].as_matrix())
        alsX = numpy.array(docX)
        alsY = numpy.array(docY)
        return alsX, alsY

    flow = (list(range(1,10,1)) + list(range(10,1,-1)))*100
    pdata = pd.DataFrame({"a":flow})
    #pdata.b = pdata.b.shift(9)
    data = pdata.iloc[10:] * random()  # some noise
    ntrn = int(round(len(data) * (1 - test_size)))
    X_train, y_train = _load_data(data.iloc[0:ntrn])
    X_test, y_test = _load_data(data.iloc[ntrn:])

    n_samples = len(X_train)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [X_train[s] for s in sidx[n_train:]]
    valid_set_y = [y_train[s] for s in sidx[n_train:]]
    train_set_x = [X_train[s] for s in sidx[:n_train]]
    train_set_y = [y_train[s] for s in sidx[:n_train]]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (X_test, y_test)
    return train, valid, test

def prepare_data(seqs, labels, maxlen=None):
    return numpy.transpose(seqs,(1,0,2)), labels

def plot_test(testy, targety):
    from matplotlib import pyplot as plt
    plt.plot(range(0,50), testy[0:50,0])
    plt.plot(range(0,50), targety[0:50,0])
    #plt.plot(range(0,50), testy[0:50,1])
    #plt.plot(range(0,50), targety[0:50,1])
    plt.show()
    

def train_lstm(dim_input=10, dim_output=2, dim_inout=2, patience=10, max_epochs=5000, dispFreq=10,
        lrate=0.0001, optimizer=adadelta, saveto='lstm_model.npz', decay_c=0.0002, validFreq=200, saveFreq=1110, batch_size=16, valid_batch_size=64):
    
    model_options = locals().copy()
    train, valid, test = load_data()
    ydim = 2

    model_options['ydim'] = dim_output

    print('Building model')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)
    tparams = init_tparams(params)
    #(x, mask, y, f_pred_prob, cost) = build_model(tparams, model_options)
    (x, y, f_pred_prob, cost) = build_model(tparams, model_options)
    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += ((tparams['U'] ** 2).sum() + 
                (tparams['U_input'] ** 2).sum())
        weight_decay *= decay_c
        cost += weight_decay
    
    #f_cost = theano.function([x, mask, y], cost, name='f_cost')
    f_cost = theano.function([x, y], cost, name='f_cost')
    grads = tensor.grad(cost, wrt=list(tparams.values()))
    #f_grad = theano.function([x, mask, y], grads, name='f_grad')
    f_grad = theano.function([x, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    #f_grad_shared, f_update = optimizer(lr, tparams, grads,
    #                                    x, mask, y, cost)
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, y, cost)
    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1

                # Select the random examples for this minibatch
                y = numpy.array([train[1][t] for t in train_index])
                x = numpy.array([train[0][t]for t in train_index])
                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x,y = prepare_data(x, y)
                n_samples += x.shape[1]
                #cost = f_grad_shared(x, mask, y)
                cost = f_grad_shared(x, y)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print('Saving...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print('Done')

                if numpy.mod(uidx, validFreq) == 0:
                    train_err = pred_error(f_pred_prob, prepare_data, train, kf)
                    valid_err = pred_error(f_pred_prob, prepare_data, valid,
                                           kf_valid)
                    test_err = pred_error(f_pred_prob, prepare_data, test, kf_test)

                    history_errs.append([valid_err, test_err])

                    if (best_p is None or
                        valid_err <= numpy.array(history_errs)[:,
                                                               0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    print( ('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err) )

                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            estop = True
                            break

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred_prob, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred_prob, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred_prob, prepare_data, test, kf_test)
    test_x, test_y = prepare_data(numpy.array(test[0]),numpy.array(test[1]))
    print(test_x)
    test_preds = f_pred_prob(test_x)
    plot_test(test_preds, test_y)
    
    print( 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err )
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print( ('Training took %.1fs' %
            (end_time - start_time)), file=sys.stderr)
    return train_err, valid_err, test_err

if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    train_lstm(
        dim_input = 50,
        dim_output = 1,
        dim_inout=1,
        max_epochs=200,
        batch_size=80,
        lrate=0.0005,
        optimizer=adadelta
    )
