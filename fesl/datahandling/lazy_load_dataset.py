import torch
from torch.utils.data import Dataset
from fesl.datahandling.snapshot import Snapshot
import numpy as np
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class.
    pass

class LazyLoadDataset(torch.utils.data.Dataset):
    """
    DataSet class for lazy loading. Only loads snapshots in the memory that are currently being processed.
    Uses a "caching" approach of keeping the last used snapshot in memory, until values from a new ones are used.
    Therefore, shuffling at DataSampler / DataLoader level is discouraged to the point that I disabled it.
    Instead, we mix the snapshot load order here ot have some sort of mixing at all.
    """
    def __init__(self, input_dimension, output_dimension, input_data_scaler, output_data_scaler, descriptor_calculator,
                 target_calculator, grid_dimensions, grid_size, descriptors_contain_xyz, use_horovod):
        """
        Parameters
        ----------
        input_dimension : int
            Dimension of an input vector.
        output_dimension : int
            Dimension of an output vector.
        input_data_scaler : fesl.datahandling.data_scaler.DataScaler
            Used to scale the input data.
        output_data_scaler : fesl.datahandling.data_scaler.DataScaler
            Used to scale the output data.
        descriptor_calculator : fesl.descriptors.descriptor_base.DescriptorBase or derivative
            Used to do unit conversion on input data.
        target_calculator : fesl.targets.target_base.TargetBase or derivative
            Used to do unit conversion on output data.
        grid_dimensions : list
            Dimensions of the grid (x,y,z).
        grid_size : int
            Size of the grid (x*y*z), i.e. the number of datapoints per snapshot.
        descriptors_contain_xyz : bool
            If true, then it is assumed that the first three entries of any input data file are xyz-information and can be discarded.
            Generally true, if your descriptors were calculated using FESL.
        use_horovod : bool
            If true, it is assumed that horovod is used.
        """
        self.snapshot_list = []
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.input_data_scaler = input_data_scaler
        self.output_data_scaler = output_data_scaler
        self.descriptor_calculator = descriptor_calculator
        self.target_calculator = target_calculator
        self.grid_dimensions = grid_dimensions
        self.grid_size = grid_size
        self.number_of_snapshots = 0
        self.total_size = 0
        self.descriptors_contain_xyz = descriptors_contain_xyz
        self.currently_loaded_file = None
        self.input_data = np.empty(0)
        self.output_data = np.empty(0)
        self.use_horovod = use_horovod

    def add_snapshot_to_dataset(self, snapshot: Snapshot):
        """
        Adds a snapshot to a DataSet. Afterwards, the DataSet can and will load this snapshot as needed.
        Parameters
        ----------
        snapshot : fesl.datahandling.snapshot.Snapshot
            Snapshot that is to be added to this DataSet.
        Returns
        -------
        """
        self.snapshot_list.append(snapshot)
        self.number_of_snapshots += 1
        self.total_size = self.number_of_snapshots*self.grid_size

    def mix_datasets(self):
        """
        Mixes the order of the snapshots so that there can be some variance between runs.
        Returns
        -------
        """
        used_perm = torch.randperm(self.number_of_snapshots)
        if self.use_horovod:
            hvd.allreduce(torch.tensor(0), name='barrier')
            used_perm = hvd.broadcast(used_perm, 0)
        self.snapshot_list = [self.snapshot_list[i] for i in used_perm]
        self.get_new_data(0)


    def get_new_data(self, file_index):
        """
        Reads new snapshot into RAM.
        Parameters
        ----------
        file_index : i
            File to be read.
        Returns
        -------
        """
        # Load the data into RAM.
        self.input_data = np.load(self.snapshot_list[file_index].input_npy_directory+self.snapshot_list[file_index].input_npy_file)
        self.output_data = np.load(self.snapshot_list[file_index].output_npy_directory+self.snapshot_list[file_index].output_npy_file)

        # Transform the data.
        if self.descriptors_contain_xyz:
            self.input_data = self.input_data[:, :, :, 3:]
        self.input_data = self.input_data.reshape([self.grid_size, self.input_dimension])
        self.input_data *= self.descriptor_calculator.convert_units(1, self.snapshot_list[file_index].input_units)
        self.input_data = self.input_data.astype(np.float32)
        self.input_data = torch.from_numpy(self.input_data).float()
        self.input_data = self.input_data_scaler.transform(self.input_data)

        self.output_data = self.output_data.reshape([self.grid_size, self.output_dimension])
        self.output_data *= self.target_calculator.convert_units(1, self.snapshot_list[file_index].output_units)
        self.output_data = np.array(self.output_data)
        self.output_data = self.output_data.astype(np.float32)
        self.output_data = torch.from_numpy(self.output_data).float()
        self.output_data = self.output_data_scaler.transform(self.output_data)

        # Save which data we have currently loaded.
        self.currently_loaded_file = file_index

    def __getitem__(self, idx):
        """
        Gets an item of the DataSet.
        Parameters
        ----------
        idx : int
            Requested index. NOTE: Slices are currently NOT supported.
        Returns
        -------
        inputs, outputs : torch.Tensor
            The requested inputs and outputs
        """
        # Get item can be called with an int or a slice.
        file_index = idx // self.grid_size
        index_in_file = idx % self.grid_size
        if file_index != self.currently_loaded_file:
            self.get_new_data(file_index)
        return self.input_data[index_in_file], self.output_data[index_in_file]

    def __len__(self):
        """
        Gets the length of the DataSet.
        Returns
        -------
        length : int
            Number of data points in DataSet.
        """
        return self.total_size