"""Runner class for running models."""
from mala.models.approx_gaussian_processes import ApproxGaussianProcesses
import torch
import gpytorch
from mala.common.printout import printout
from mala.common.parameters import ParametersRunning
from mala.models.gaussian_processes import GaussianProcesses
from mala import Parameters
import numpy as np
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class
    pass


class Runner:
    """
    Parent class for all classes that in some sense "run" the models.

    That can be training, benchmarking, inference, etc.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this Runner object.

    model : mala.model.model.Model
        Network which is being run.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler holding the data for the run.
    """

    def __init__(self, params, model, data):
        self.parameters_full: Parameters = params
        self.parameters: ParametersRunning = params.running
        self.model = model
        self.data = data
        self.gaussian_processes_used = False
        self.approx_gaussian_processes_used = False
        if isinstance(self.model, GaussianProcesses):
            self.gaussian_processes_used = True
        if self.gaussian_processes_used and self.parameters.mini_batch_size \
                != 1:
            printout("Adjusting mini batch size because Gaussian processes are "
                     "being used.")
            self.parameters.mini_batch_size = 1
        if isinstance(self.model, ApproxGaussianProcesses):
            self.approx_gaussian_processes_used = True
        self.__prepare_to_run()
        
    def __prepare_to_run(self):
        """
        Prepare the Runner to run the Network.

        This includes e.g. horovod setup.
        """
        # See if we want to use horovod.
        if self.parameters_full.use_horovod:
            if self.parameters_full.use_gpu:
                printout("size=", hvd.size(), "global_rank=", hvd.rank(),
                         "local_rank=", hvd.local_rank(), "device=",
                         torch.cuda.get_device_name(hvd.local_rank()))
                # pin GPU to local rank
                torch.cuda.set_device(hvd.local_rank())

    def _predict_from_array(self, data_set, 
                            number_of_batches_per_snapshot=0,
                            batch_size=0):
        
        predicted_outputs = np.zeros((len(data_set),
                                      self.data.get_output_dimension()))

        if not self.gaussian_processes_used:
            for i in range(0, number_of_batches_per_snapshot):
                inputs = self.data.input_data_scaler.transform(
                    data_set[(i * batch_size):((i + 1) * batch_size)])
                if self.parameters_full.use_gpu:
                    inputs = inputs.to('cuda')
                if self.approx_gaussian_processes_used:
                    if self.parameters_full.use_multitask_gp:
                        predicted_outputs[i * batch_size:(i + 1) * batch_size, :] = \
                                                self.data.output_data_scaler.\
                                                inverse_transform(self.model.likelihood(
                                                self.model(inputs)).mean.to('cpu'),
                                                as_numpy=True)
                    else:
                        out = self.data.output_data_scaler.\
                            inverse_transform(self.model(inputs).mean.
                                            to('cpu'), as_numpy=True)
                        predicted_outputs[i * batch_size:(i + 1) * batch_size, :] = \
                            np.reshape(out, (-1, 1))
                    
                else:
                    predicted_outputs[i * batch_size:(i + 1) * batch_size, :] = \
                        self.data.output_data_scaler.\
                        inverse_transform(self.model(inputs).
                                        to('cpu'), as_numpy=True)
        else:
            inputs = self.data.input_data_scaler.transform(data_set)
            if self.parameters_full.use_gpu:
                inputs = inputs.to('cuda')
            with torch.no_grad():
                if self.parameters_full.use_fast_pred_var:
                    with gpytorch.settings.fast_pred_var():
                        predicted_outputs = self.data.output_data_scaler.\
                                                inverse_transform(self.model.likelihood(
                                                self.model(inputs)).mean.to('cpu'),
                                                as_numpy=True)
                else:
                    predicted_outputs = self.data.output_data_scaler.\
                                            inverse_transform(self.model.likelihood(
                                            self.model(inputs)).mean.to('cpu'),
                                            as_numpy=True)
            
            if self.parameters_full.use_multitask_gp:
                print(predicted_outputs.shape)
                predicted_outputs = predicted_outputs.reshape(-1, self.parameters_full.model.no_of_tasks)
                print(predicted_outputs.shape)
            else:
                predicted_outputs = predicted_outputs.transpose(1, 0)

        # Restricting the actual quantities to physical meaningful values,
        # i.e. restricting the (L)DOS to positive values.
        if self.parameters_full.targets.target_type == "Energy density":
            predicted_outputs[:,0] = self.data.target_calculator.\
            restrict_data(predicted_outputs[:,0])
        else:
            predicted_outputs = self.data.target_calculator.\
                restrict_data(predicted_outputs)

        # It could be that other operations will be happening with the data
        # set, so it's best to reset it.
        if self.data.parameters.use_lazy_loading:
            data_set.return_outputs_directly = False

        return predicted_outputs


    def _forward_entire_snapshot(self, snapshot_number, data_set,
                                 number_of_batches_per_snapshot=0,
                                 batch_size=0):
        """
        Forward a snapshot through the models, get actual/predicted output.

        Parameters
        ----------
        snapshot_number : int
            Snapshot for which the prediction is done.

        number_of_batches_per_snapshot : int
            Number of batches that lie within a snapshot.

        batch_size : int
            Batch size used for forward pass.

        Returns
        -------
        actual_outputs : torch.Tensor
            Actual outputs for snapshot.

        predicted_outputs : torch.Tensor
            Precicted outputs for snapshot.
        """
        if self.data.parameters.use_lazy_loading:
            data_set.return_outputs_directly = True
            actual_outputs = \
                (data_set
                 [snapshot_number * self.data.
                    grid_size:(snapshot_number + 1) * self.data.grid_size])[1].\
                    detach().numpy().astype(np.float64)
        else:
            actual_outputs = \
                self.data.output_data_scaler.\
                inverse_transform(
                    (data_set[snapshot_number * self.data.grid_size:
                    (snapshot_number + 1) * self.data.grid_size])[1],
                    as_numpy=True)

        predicted_outputs = np.zeros((self.data.grid_size,
                                      self.data.get_output_dimension()))

        offset = snapshot_number * self.data.grid_size
        if not self.gaussian_processes_used:
            for i in range(0, number_of_batches_per_snapshot):
                inputs, outputs = \
                    data_set[offset+(i * batch_size):offset+((i + 1) * batch_size)]
                if self.parameters_full.use_gpu:
                    inputs = inputs.to('cuda')
                if self.approx_gaussian_processes_used:
                    if self.parameters_full.use_multitask_gp:
                        predicted_outputs[i * batch_size:(i + 1) * batch_size, :] = \
                                                self.data.output_data_scaler.\
                                                inverse_transform(self.model.likelihood(
                                                self.model(inputs)).mean.to('cpu'),
                                                as_numpy=True)
                    else:
                        out = self.data.output_data_scaler.\
                            inverse_transform(self.model(inputs).mean.
                                            to('cpu'), as_numpy=True)
                        predicted_outputs[i * batch_size:(i + 1) * batch_size, :] = \
                            np.reshape(out, (-1, 1))
                    
                else:
                    predicted_outputs[i * batch_size:(i + 1) * batch_size, :] = \
                        self.data.output_data_scaler.\
                        inverse_transform(self.model(inputs).
                                        to('cpu'), as_numpy=True)
        else:
            inputs = \
                    data_set[snapshot_number * self.data.grid_size: (snapshot_number + 1) * self.data.grid_size][0]
            if self.parameters_full.use_gpu:
                inputs = inputs.to('cuda')
            with torch.no_grad():
                if self.parameters_full.use_fast_pred_var:
                    with gpytorch.settings.fast_pred_var():
                        predicted_outputs = self.data.output_data_scaler.\
                                                inverse_transform(self.model.likelihood(
                                                self.model(inputs)).mean.to('cpu'),
                                                as_numpy=True)
                else:
                    predicted_outputs = self.data.output_data_scaler.\
                                            inverse_transform(self.model.likelihood(
                                            self.model(inputs)).mean.to('cpu'),
                                            as_numpy=True)
            
            if self.parameters_full.use_multitask_gp:
                print(predicted_outputs.shape)
                print(actual_outputs.shape)
                predicted_outputs = predicted_outputs.reshape(-1, self.parameters_full.model.no_of_tasks)
                actual_outputs = actual_outputs.reshape(-1, self.parameters_full.model.no_of_tasks)
                print(predicted_outputs.shape)
                print(actual_outputs.shape)
            else:
                predicted_outputs = predicted_outputs.transpose(1, 0)
                actual_outputs = actual_outputs.transpose(1, 0)

        # Restricting the actual quantities to physical meaningful values,
        # i.e. restricting the (L)DOS to positive values.
        if self.parameters_full.targets.target_type == "Energy density":
            predicted_outputs[:,0] = self.data.target_calculator.\
            restrict_data(predicted_outputs[:,0])
        else:
            predicted_outputs = self.data.target_calculator.\
                restrict_data(predicted_outputs)

        # It could be that other operations will be happening with the data
        # set, so it's best to reset it.
        if self.data.parameters.use_lazy_loading:
            data_set.return_outputs_directly = False

        return actual_outputs, predicted_outputs

    @staticmethod
    def _correct_batch_size_for_testing(datasize, batchsize):
        """
        Get the correct batch size for testing.

        For testing snapshot the batch size needs to be such that
        data_per_snapshot / batch_size will result in an integer division
        without any residual value.
        """
        new_batch_size = batchsize
        if datasize % new_batch_size != 0:
            while datasize % new_batch_size != 0:
                new_batch_size += 1
        return new_batch_size
