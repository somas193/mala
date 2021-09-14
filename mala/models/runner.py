"""Runner class for running models."""
import torch
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
        if isinstance(self.parameters_full.model, GaussianProcesses):
            printout("Adjusting mini batch size because Gaussian processes are "
                     "being used.")
            self.parameters.mini_batch_size = 1
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
                     grid_size:(snapshot_number + 1) * self.data.grid_size])[1]
        else:
            actual_outputs = \
                self.data.output_data_scaler.\
                inverse_transform(
                    (data_set[snapshot_number *
                                             self.data.grid_size:
                                             (snapshot_number + 1) *
                                             self.data.grid_size])[1],
                    as_numpy=True)

        predicted_outputs = np.zeros((self.data.grid_size,
                                      self.data.get_output_dimension()))

        offset = snapshot_number * self.data.grid_size
        for i in range(0, number_of_batches_per_snapshot):
            inputs, outputs = \
                data_set[offset+(i * batch_size):offset+((i + 1) * batch_size)]
            if self.parameters_full.use_gpu:
                inputs = inputs.to('cuda')
            predicted_outputs[i * batch_size:(i + 1) * batch_size, :] = \
                self.data.output_data_scaler.\
                inverse_transform(self.model(inputs).
                                  to('cpu'), as_numpy=True)

        # Restricting the actual quantities to physical meaningful values,
        # i.e. restricting the (L)DOS to positive values.
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
