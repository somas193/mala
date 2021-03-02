"""Objective functions for hyperparameter optimizations without training."""
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from fesl.common.parameters import Parameters
from fesl.datahandling.data_handler import DataHandler
from fesl.network.network import Network
from fesl.network.objective_base import ObjectiveBase
from fesl.common.parameters import printout
import matplotlib.pyplot as plt


class ObjectiveNoTraining(ObjectiveBase):
    """
    Represents the objective function using no NN training.

    The objective value is calculated using the Jacobian.
    """

    def __init__(self, search_parameters: Parameters, data_handler:
                 DataHandler, trial_type):
        """
        Create an ObjectiveNoTraining object.
        
        Parameters
        ----------
        search_parameters : fesl.common.parameters.Parameters
            Parameters used to create this objective.
            
        data_handler : fesl.datahandling.data_handler.DataHandler
            datahandler to be used during the hyperparameter optimization.
        
        trial_type : string
            Format of hyperparameters used in this objective. Supported 
            choices are:
            
                - optuna
                - oat
        """
        super(ObjectiveNoTraining, self).__init__(search_parameters,
                                                  data_handler)
        self.trial_type = trial_type

    def __call__(self, trial):
        """
        Get objective value for a trial (=set of hyperparameters).

        Parameters
        ----------
        trial
            A trial is a set of hyperparameters; can be an optuna based
            trial or simply a OAT compatible list.
        """
        # Parse the parameters using the base class.
        super(ObjectiveNoTraining, self).parse_trial(trial)

        # Build the network.
        net = Network(self.params)
        device = "cuda" if self.params.use_gpu else "cpu"

        # Load the batchesand get the jacobian.
        loader = DataLoader(self.data_handler.training_data_set,
                            batch_size=self.params.running.mini_batch_size,
                            shuffle=True)
        jac = ObjectiveNoTraining.__get_batch_jacobian(net, loader, device)

        # Loss = - score!
        surrogate_loss = float('inf')
        try:
            surrogate_loss = - ObjectiveNoTraining.__calc_score(jac)
            surrogate_loss = surrogate_loss.detach().numpy().astype(np.float64)
        except RuntimeError:
            printout("Got a NaN, ignoring sample.")
        return surrogate_loss

    @staticmethod
    def __get_batch_jacobian(net: Network, loader: DataLoader, device) \
            -> Tensor:
        """Calculate the jacobian of the batch."""
        x: Tensor
        (x, _) = next(iter(loader))
        x = x.to(device)
        net.zero_grad()
        x.requires_grad_(True)
        y: Tensor = net(x)
        y.backward(torch.ones_like(y))
        jacobian = x.grad.detach()
        return jacobian

    @staticmethod
    def __corrcoef(x):
        """
        Calculate the correlation coefficient of an array.

        Pytorch implementation of np.corrcoef.
        """
        mean_x = torch.mean(x, 1, True)
        xm = x.sub(mean_x.expand_as(x))
        c = xm.mm(xm.t())
        c = c / (x.size(1) - 1)

        # normalize covariance matrix
        d = torch.diag(c)
        stddev = torch.pow(d, 0.5)
        c = c.div(stddev.expand_as(c))
        c = c.div(stddev.expand_as(c).t())

        # clamp between -1 and 1
        # probably not necessary but numpy does it
        c = torch.clamp(c, -1.0, 1.0)

        return c

    @staticmethod
    def __calc_score(jacobian: Tensor):
        """Calculate the score for a Jacobian."""
        correlations = ObjectiveNoTraining.__corrcoef(jacobian)
        eigen_values, _ = torch.eig(correlations)
        # Only consider the real valued part, imaginary part is rounding
        # artefact
        eigen_values = eigen_values.type(torch.float32)
        # Needed for numerical stability. 1e-4 instead of 1e-5 in reference
        # as the torchs eigenvalue solver on GPU
        # seems to have bigger rounding errors than numpy, resulting in
        # slightly larger negative Eigenvalues
        k = 1e-4
        v = -torch.sum(torch.log(eigen_values + k) + 1. / (eigen_values+k))
        return v
