import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import torch
from GP import ModelFactory
import os

import time
# def cost_time(func):
#     def fun(*args, **kwargs):
#         t = time.perf_counter()
#         result = func(*args, **kwargs)
#         print(f'func {func.__name__} coast time:{time.perf_counter() - t:.8f} s')
#         return result
#
#     return fun


mf = ModelFactory()
class Acquisition(object):
    def __init__(self, function):
        self._function = function
    def function(self, X, X_sample, Y_sample, gpr):
        return self._function( X, X_sample, Y_sample, gpr)


    def next_sample(self, X_sample, Y_sample, gpr, bounds, n_restarts= 25, Noise = True):
        dim = X_sample.shape[1]
        min_val = 1e9
        min_x = None

        #mu_sample = mf.get_output(X_sample.cuda(), gpr, return_std=False)
        def min_obj(X):
            return -self.function(X.reshape(-1, dim), X_sample, Y_sample, gpr)

        #min_obj = lambda X : -self.function(X.reshape(-1, dim), X_sample, Y_sample, gpr)
        # def min_obj(X):
        #     # Minimization objective is the negative acquisition function
        #     res = -fun(X)
        #     return -fun(X)

            # Find the best optimum by starting from n_restart different random points.

        for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds=None, method='L-BFGS-B')
            #res = min(min_obj)
            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x

        return min_x.reshape(-1, dim)

def expected_improvement(X, X_sample, Y_sample, gpr, trade_off, Noise = True):
    mu, sigma = mf.get_output(torch.tensor(X).cuda(), gpr, return_std=True)
    mu_sample = mf.get_output(X_sample.cuda(), gpr, return_std=False)

    sigma = sigma.reshape(-1, 1)

    # otherwise use np.max(Y_sample).
    if Noise:
        mu_sample_opt = np.max(mu_sample)
    else:
        mu_sample_opt = np.max(Y_sample)

    with np.errstate(divide='warn'):
        imp = mu_sample_opt - mu - trade_off
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei[0]

class EI(Acquisition):
    def __init__(self, trade_off = 0.01, Noise = True):


        function = lambda X, X_sample, Y_sample, gpr: expected_improvement(X, X_sample, Y_sample, gpr, trade_off, Noise)

        try:
            super().__init__(function)
        except:
            super(EI, self).__init__(function)

