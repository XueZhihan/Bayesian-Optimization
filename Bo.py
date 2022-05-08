import numpy as np
import torch
from Test_functions import Function
from acquisition import Acquisition
from GP import ModelFactory
import gpytorch
import time
from scipy.optimize import minimize
from acquisition import expected_improvement
from matplotlib import pyplot as plt

class BO_Trainer():
    def __init__(self, test_function: Function, acquisition: Acquisition, likelihood, model_name):
        self._test_function = test_function
        self.model_name = model_name
        self.gp_model = None
        self._likelihood = likelihood
        self._acquisition = acquisition
    @property
    def test_function(self):
        return self._test_function

    @property
    def likelihood(self):
        return self._likelihood

    @property
    def acquisition(self):
        return self._acquisition

    def generate_model(self, X_sample, Y_sample):

        mf = ModelFactory()
        self.gp_model = mf.creat_model(self.model_name, X_sample, Y_sample, self.likelihood).cuda()
        training_iter = 50

        # Find optimal model hyperparameters
        self.gp_model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
        # self.gp_model.eval()
        # self.likelihood.eval()
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.gp_model(X_sample)
            # Calc loss and backprop gradients
            loss = -mll(output, Y_sample)
            loss.backward()
            optimizer.step()
        return self.gp_model

    def train(self, test_function, experiments, n_iter, verbose=False):
        Error_gaps_list = []

        for i in range(experiments):
            print(f"\nExperiments: {i + 1} \t", end="")

            train_x = test_function.sample(20)
            train_y = test_function.output(train_x)

            train_x = torch.tensor(train_x).cuda()
            train_y = torch.tensor(train_y).cuda()

            X_sample = train_x
            Y_sample = train_y

            Error_gaps = []

            for j in range(n_iter):
                # Update Gaussian process with existing samples
                gpr = self.generate_model(train_x, train_y)

                # Obtain next sampling point from the acquisition function (expected_improvement)
                X_next = self.acquisition.next_sample(X_sample, Y_sample, gpr, self.test_function.get_bounds())

                # Obtain next noisy sample from the objective function
                Y_next = torch.tensor(self.test_function.output(X_next))

                # Add sample to previous samples
                X_sample = torch.tensor(np.vstack((X_sample.cpu(), X_next)))
                Y_sample = torch.tensor(np.concatenate((Y_sample.cpu(), Y_next)))
                min_f_star = self.test_function.get_global_minimum()
                Error_gaps.append(np.abs(min_f_star - min(Y_sample).item()))
                if verbose:
                    print(f"\n\tIter {n_iter} / {j + 1} - Error gap: {Error_gaps[-1]:.5f}", end="")
                else:
                    if j % 100 == 0:
                        print()
                    print(".", end="")

            Error_gaps_list.append(Error_gaps)
        plot(Error_gaps_list, n_iter, experiments)
        return Error_gaps_list



def plot(gaps, n_iter, experiment):
    def ci(y):
        # calculate confidence interval
        return 1.96 * y.std(axis=0) / np.sqrt(experiment)

    iters = np.arange(n_iter)

    # calculate error gap.
    error_gap_f = np.asarray(gaps)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.errorbar(iters, error_gap_f.mean(axis=0), yerr=ci(error_gap_f), label="FixedNoiseGaussianLikelihood", linewidth=1.5)

    ax.set_ylim(bottom=-0.1)
    ax.set(xlabel='number of observations (beyond initial points)', ylabel='error gap')
    ax.legend(loc="upper right")
    plt.savefig('./results/result.jpg')
    plt.show()


