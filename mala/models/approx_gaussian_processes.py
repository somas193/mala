import gpytorch
import torch

from mala import DataHandler
from mala.common.parameters import ParametersModel

class ApproxGaussianProcesses(gpytorch.models.ApproximateGP):
    def __init__(self, params, inducing_points):
        self.params: ParametersModel = params.model

        # if the user has planted a seed (for comparibility purposes) we
        # should use it.
        if params.manual_seed is not None:
            torch.manual_seed(params.manual_seed)
            torch.cuda.manual_seed(params.manual_seed)

        # Parse parameters.

        # Variational distribution.
        variational_distribution = None
        if self.params.variational_dist_type == "cholesky":
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        if self.params.variational_dist_type == "mean_field":
            variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(inducing_points.size(0))
        
        if variational_distribution is None:
            raise Exception("Invalid Variational distribution selected.")

        # Variational strategy.
        variational_strat = None
        if self.params.variational_strategy_type == "variational_strategy":
            variational_strat = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)

        if variational_strat is None:
            raise Exception("Invalid Variational strategy selected.")

        super(ApproxGaussianProcesses, self).__init__(variational_strat)

        # Likelihood.
        self.likelihood = None
        if self.params.loss_function_type == "gaussian_likelihood":
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        if self.likelihood is None:
            raise Exception("Invalid Likelihood selected.")

        # Mean.
        self.mean_module = None
        if self.params.gp_mean == "constant":
            self.mean_module = gpytorch.means.ConstantMean()

        if self.mean_module is None:
            raise Exception("Invalid mean module selected.")

        # Kernel.
        self.covar_module = None
        if self.params.kernel == "rbf":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        if self.params.kernel == "linear":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        if self.params.kernel == "matern":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())

        if self.covar_module is None:
            raise Exception("Invalid kernel selected.")

        # Multivariate distribution.
        self.multivariate_distribution = None
        if self.params.multivariate_distribution == "normal":
            self.multivariate_distribution = gpytorch.distributions.MultivariateNormal

        if self.multivariate_distribution is None:
            raise Exception("Invalid multivariate distribution selected.")
        
        # Once everything is done, we can move the Network on the target
        # device.
        if params.use_gpu:
            self.likelihood.to('cuda')
            self.to('cuda')
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return self.multivariate_distribution(mean_x, covar_x)

    def calculate_loss(self, output, target):
        if self.params.max_log_likelihood == "elbo":
            return -gpytorch.mlls.VariationalELBO(self.likelihood, self, num_data=target.size(0))(output, target)

        if self.params.max_log_likelihood == "pll":
            return -gpytorch.mlls.PredictiveLogLikelihood(self.likelihood, self, num_data=target.size(0))(output, target)
        

    # TODO: implement this.
    def load_from_file(cls, params, path_to_file):
        pass

    # TODO: Implement this.
    def save_model(self, path_to_file):
        pass

    def train(self, mode=True):
        self.likelihood.train(mode=mode)
        return super(ApproxGaussianProcesses, self).train(mode=mode)

    def eval(self):
        self.likelihood.eval()
        return super(ApproxGaussianProcesses, self).eval()
        
