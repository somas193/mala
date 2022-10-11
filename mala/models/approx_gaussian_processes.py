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
            if params.use_multitask_gp:
                variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(1), 
                                           batch_shape=torch.Size([self.params.no_of_latents]))
            else:
                variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        if self.params.variational_dist_type == "mean_field":
            if params.use_multitask_gp:
                variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(inducing_points.size(1), 
                                           batch_shape=torch.Size([self.params.no_of_latents]))
            else:
                variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(inducing_points.size(0))
        if self.params.variational_dist_type == "delta":
            if params.use_multitask_gp:
                variational_distribution = gpytorch.variational.DeltaVariationalDistribution(inducing_points.size(1), 
                                           batch_shape=torch.Size([self.params.no_of_latents]))
            else:
                variational_distribution = gpytorch.variational.DeltaVariationalDistribution(inducing_points.size(0))
        
        if variational_distribution is None:
            raise Exception("Invalid Variational distribution selected.")

        # Variational strategy.
        variational_strat = None
        if self.params.variational_strategy_type == "LMC" and (self.params.no_of_tasks > 1):
            variational_strat = gpytorch.variational.LMCVariationalStrategy(gpytorch.variational.VariationalStrategy(self, 
                                inducing_points, variational_distribution, learn_inducing_locations=True), num_tasks=self.params.no_of_tasks, 
                                num_latents=self.params.no_of_latents, latent_dim=-1)

        if self.params.variational_strategy_type == "variational_strategy":
            variational_strat = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, 
                                learn_inducing_locations=True)
        if variational_strat is None:
            raise Exception("Invalid Variational strategy selected.")

        super(ApproxGaussianProcesses, self).__init__(variational_strat)

        # Likelihood.
        self.likelihood = None
        if self.params.loss_function_type == "gaussian_likelihood":
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if self.params.loss_function_type == "multitask":
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.params.no_of_tasks)

        if self.likelihood is None:
            raise Exception("Invalid Likelihood selected.")

        # Mean.
        self.mean_module = None
        if self.params.gp_mean == "constant":
            if params.use_multitask_gp:
                self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([self.params.no_of_latents]))
            else:
                self.mean_module = gpytorch.means.ConstantMean()
        if self.params.gp_mean == "linear":
            if params.use_multitask_gp:
                self.mean_module = gpytorch.means.LinearMean(batch_shape=torch.Size([self.params.no_of_latents]))
            else:
                self.mean_module = gpytorch.means.LinearMean(1)
        if self.mean_module is None:
            raise Exception("Invalid mean module selected.")

        # Kernel.
        self.covar_module = None
        if self.params.kernel == "rbf":
            if params.use_multitask_gp:
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(batch_shape=torch.Size([self.params.no_of_latents])), 
                                    batch_shape=torch.Size([self.params.no_of_latents]))
            else:    
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        if self.params.kernel == "linear":
            if params.use_multitask_gp:
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel(batch_shape=torch.Size([self.params.no_of_latents])), 
                                    batch_shape=torch.Size([self.params.no_of_latents]))
            else:    
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        if self.params.kernel == "matern":
            if params.use_multitask_gp:
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, batch_shape=torch.Size([self.params.no_of_latents])), 
                                    batch_shape=torch.Size([self.params.no_of_latents]))
            else:    
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=91))
        if self.params.kernel == "polynomial":
            if params.use_multitask_gp:
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=2, batch_shape=torch.Size([self.params.no_of_latents])), 
                                    batch_shape=torch.Size([self.params.no_of_latents]))
            else:    
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=3))


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

    def calculate_loss(self, output, target, nr_train_data):
        if self.params.max_log_likelihood == "elbo":
            return -gpytorch.mlls.VariationalELBO(self.likelihood, self, num_data=nr_train_data)(output, target)

        if self.params.max_log_likelihood == "pll":
            return -gpytorch.mlls.PredictiveLogLikelihood(self.likelihood, self, num_data=nr_train_data)(output, target)
        

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
        
