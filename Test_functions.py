
import numpy as np


class Function(object):
    def __init__(self, dimensionality, bounds, global_minimizers, global_minimum, function):
        self._dimensionality = dimensionality
        self._bounds = bounds
        self._global_minimizers = global_minimizers
        self._global_minimum = global_minimum
        self._function = function

    @property
    def bounds(self):
        return self._bounds

    @property
    def dimensionality(self):
        return self._dimensionality

    @property
    def global_minimizers(self):
        return self._global_minimizers

    @property
    def global_minimum(self):
        return self._global_minimum

    def get_bounds(self):
        return self.bounds

    def get_global_minimizers(self):
        return self.global_minimizers

    def get_global_minimum(self):
        return self.global_minimum

    def function(self, bx):
        return self._function(bx)

    def _output(self, X):
        bounds = self.get_bounds()

        if len(X.shape) == 2:
            list_results = [self.function(bx) for bx in X]
        else:
            list_results = [self.function(X)]

        by = np.array(list_results)
        return by

    def output(self, X):
        by = self._output(X)
        #Y = np.expand_dims(by, axis=1)

        return by

    def sample(self, num_points, seed=None):

        random_state_ = np.random.RandomState(seed)

        bounds = self.get_bounds()

        points = random_state_.uniform(size=(num_points, self.dimensionality))
        points = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * points

        return points

def fun_Hartmann6D(bx, dim_bx):

    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])

    outer = 0.0
    for i_ in range(0, 4):
        inner = 0.0
        for j_ in range(0, 6):
            inner += A[i_, j_] * (bx[j_] - P[i_, j_])**2
        outer += alpha[i_] * np.exp(-1.0 * inner)

    y = -1.0 * outer
    return y


class Hartmann6D(Function):
    def __init__(self,
        bounds=np.array([
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]),
    ):

        dim_bx = 6
        assert bounds.shape[0] == dim_bx

        global_minimizers = np.array([
            [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573],
        ])
        global_minimum = -3.322368
        function = lambda bx: fun_Hartmann6D(bx, dim_bx)

        try:
            super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function)
        except:
            super(Hartmann6D, self).__init__(dim_bx, bounds, global_minimizers, global_minimum, function)