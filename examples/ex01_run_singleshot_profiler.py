import mala
from mala import printout
from data_repo_path import data_repo_path
data_path = data_repo_path + "Be2/densities_gp/"

from pathlib import Path

Path("./traces").mkdir(parents=True, exist_ok=True)
import random
import numpy
import torch
from torch.profiler import profile, record_function, ProfilerActivity


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 14012022
seed_all(seed)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True,
             profile_memory=True,
             use_cuda=True) as prof:
    """
    ex01_run_singleshot.py: Shows how a neural models can be trained on material 
    data using this framework. It uses preprocessed data, that is read in 
    from *.npy files.
    """

    ####################
    # PARAMETERS
    # All parameters are handled from a central parameters class that
    # contains subclasses.
    ####################

    test_parameters = mala.Parameters()
    test_parameters.use_gpu = True
    test_parameters.manual_seed = seed
    # Currently, the splitting in training, validation and test set are
    # done on a "by snapshot" basis. Specify how this is
    # done by providing a list containing entries of the form
    # "tr", "va" and "te".
    test_parameters.data.data_splitting_type = "by_snapshot"

    # Specify the data scaling.
    test_parameters.data.input_rescaling_type = "feature-wise-standard"
    test_parameters.data.output_rescaling_type = "normal"

    # Specify the used activation function.
    test_parameters.model.layer_activations = ["ReLU"]

    # Specify the training parameters.
    test_parameters.running.max_number_epochs = 1
    test_parameters.running.mini_batch_size = 3000
    test_parameters.running.learning_rate = 0.00001
    test_parameters.running.trainingtype = "Adam"
    test_parameters.targets.target_type = "Density"

    with record_function("data_loader"):
        ####################
        # DATA
        # Add and prepare snapshots for training.
        ####################

        data_handler = mala.DataHandler(test_parameters)

        # Add a snapshot we want to use in to the list.
        inputs_folder = data_path + "inputs_snap/"
        outputs_folder = data_path + "outputs_density/"
        additional_folder = data_path + "additional_info_qeouts/"
        data_handler.add_snapshot("snapshot0.in.npy", inputs_folder,
                                "snapshot0.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
        data_handler.add_snapshot("snapshot1.in.npy", inputs_folder,
                                "snapshot1.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
        data_handler.add_snapshot("snapshot2.in.npy", inputs_folder,
                                "snapshot2.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
                                
        data_handler.add_snapshot("snapshot3.in.npy", inputs_folder,
                                "snapshot3.out.npy", outputs_folder, add_snapshot_as="va", output_units="None")
        data_handler.add_snapshot("snapshot4.in.npy", inputs_folder,
                                "snapshot4.out.npy", outputs_folder, add_snapshot_as="te",
                                output_units="None", calculation_output_file=additional_folder+"snapshot4.out")
        data_handler.prepare_data()

    with record_function("network_setup"):
        ####################
        # NETWORK SETUP
        # Set up the models and trainer we want to use.
        # The layer sizes can be specified before reading data,
        # but it is safer this way.
        ####################

        test_parameters.model.layer_sizes = [data_handler.get_input_dimension(),
                                             400, 800, 400,
                                             data_handler.get_output_dimension()]

        # Setup models and trainer.
        test_network = mala.Network(test_parameters)
        test_trainer = mala.Trainer(test_parameters, test_network, data_handler)

    with record_function("network_train"):
        ####################
        # TRAINING
        # Train the models.
        ####################
        test_trainer.train_model()

    test_parameters.running.mini_batch_size = 8000 # to forward the entire snap
    with record_function("tester_setup"):
        ####################
        # TESTING
        # Pass the first test set snapshot (the test snapshot).
        ####################

        tester = mala.Tester(test_parameters, test_network, data_handler)
    
    with record_function("inference"):
        actual_density, predicted_density = tester.test_snapshot(0)

    with record_function("target_calculation"):
        # First test snapshot --> 2nd in total
        data_handler.target_calculator.read_additional_calculation_data("qe.out",
                                                                        data_handler.get_snapshot_calculation_output(4))
        actual_number_of_electrons = data_handler.target_calculator.get_number_of_electrons(actual_density)
        predicted_number_of_electrons = data_handler.target_calculator.get_number_of_electrons(predicted_density)
        printout(
            f"actual_number_of_electrons: {actual_number_of_electrons}, predicted_number_of_electrons: {predicted_number_of_electrons}")

#print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=3))
#print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=3))
#print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=3))
#print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=3))

if test_parameters.use_gpu:
    dev = "gpu"
else:
    dev = "cpu"
prof.export_chrome_trace(f"./traces/NN_trace_{dev}.json")
