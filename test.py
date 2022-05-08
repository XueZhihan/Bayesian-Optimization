from Test_functions import Hartmann6D
from acquisition import EI
from GP import GPModel
from Bo import BO_Trainer
import gpytorch
import torch
import numpy as np
from matplotlib import pyplot as plt







if __name__ == '__main__':
    if torch.cuda.is_available():
        test_function = Hartmann6D()
        model_name = "ExactGP"
        acq = EI()
        noises = torch.ones(20) * 0.01
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises, learn_additional_noise=True).cuda()
        trainer = BO_Trainer(test_function,  acq, likelihood, model_name)
        trainer.train(test_function, 10, 200)



    else:
        print(torch.__version__)

        print(torch.version.cuda)
        print(torch.backends.cudnn.version())