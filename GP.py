
import numpy as np
import gpytorch
from gpytorch import kernels, means, models, mlls, settings
from gpytorch import distributions as distr
import torch

class GPModel(models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = means.ConstantMean()
        self.covar_module = kernels.ScaleKernel(kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return distr.MultivariateNormal(mean_x, covar_x)

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)






class ModelFactory:
    def creat_model(self, name, x, y, likelihood):
        if name == "ExactGP":
            return GPModel(x, y, likelihood)
        if name == "MultiTaskGP":
            return MultitaskGPModel(x, y, likelihood)
        else:
            raise TypeError("No such model %s" % name)
    def get_output(self, X, model, return_std=True):
        model.eval()
        model.likelihood.eval()
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)

        with torch.no_grad():
            pred_y = model(X.cuda())
        #pred_y = model.likelihood(pred)
        mu = pred_y.mean.detach().cpu().numpy()
        std = np.sqrt(pred_y.variance.detach().cpu().numpy())
        if return_std:
            return mu, std
        return mu






