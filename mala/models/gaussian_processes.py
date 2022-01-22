import gpytorch
import torch

from mala import DataHandler
from mala.common.parameters import ParametersModel


class GaussianProcesses(gpytorch.models.ExactGP):
    def __init__(self, params, data_handler: DataHandler, num_gpus=1):
        self.params: ParametersModel = params.model

        # if the user has planted a seed (for comparibility purposes) we
        # should use it.
        if params.manual_seed is not None:
            torch.manual_seed(params.manual_seed)
            torch.cuda.manual_seed(params.manual_seed)

        # Parse parameters.
        # Likelihood.
        likelihood = None
        if self.params.loss_function_type == "gaussian_likelihood":
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

        if likelihood is None:
            raise Exception("Invalid Likelihood selected.")

        if params.use_gpu:
            training_data_inputs = data_handler.training_data_inputs.to(torch.device('cuda:0'))
            training_data_outputs = data_handler.training_data_outputs.to(torch.device('cuda:0'))
        else:
            training_data_inputs = data_handler.training_data_inputs
            training_data_outputs = data_handler.training_data_outputs

        super(GaussianProcesses, self).__init__(training_data_inputs,
                                                training_data_outputs,
                                                likelihood)
        # Mean.
        self.mean_module = None
        if self.params.gp_mean == "constant":
            self.mean_module = gpytorch.means.ConstantMean()

        if self.mean_module is None:
            raise Exception("Invalid mean module selected.")

        # Kernel.
        self.covar_module = None
        if self.params.kernel == "rbf":
            base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        if self.params.kernel == "linear":
            base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        if self.params.kernel == "rbf+linear":
            base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel() + gpytorch.kernels.LinearKernel())
        if self.params.kernel == "polynomial":
            base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=2))
        if self.params.kernel == "cosine":
            base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())

        if params.use_gpu and (num_gpus > 1):
            self.covar_module = gpytorch.kernels.MultiDeviceKernel(base_covar_module, device_ids=range(num_gpus),
                                                                   output_device=torch.device('cuda:0'))
            print('Planning to run on {} GPUs.'.format(num_gpus))
        else:
            self.covar_module = base_covar_module

        if self.covar_module is None:
            raise Exception("Invalid kernel selectded.")

        # Multivariate distribution.
        self.multivariate_distribution = None
        if self.params.multivariate_distribution == "normal":
            self.multivariate_distribution = gpytorch.distributions.MultivariateNormal

        if self.multivariate_distribution is None:
            raise Exception("Invalid multivariate distribution selected.")
            
        # Once everything is done, we can move the Network on the target
        # device.
        if params.use_gpu:
            self.likelihood.to(torch.device('cuda:0'))
            self.to(torch.device('cuda:0'))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return self.multivariate_distribution(mean_x, covar_x)

    def calculate_loss(self, output, target):
        return -gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)(output, target)

    # TODO: implement this.
    def load_from_file(cls, params, path_to_file):
        pass

    # TODO: Implement this.
    def save_model(self, path_to_file):
        pass

    def train(self, mode=True):
        self.likelihood.train(mode=mode)
        return super(GaussianProcesses, self).train(mode=mode)

    def eval(self):
        self.likelihood.eval()
        return super(GaussianProcesses, self).eval()

