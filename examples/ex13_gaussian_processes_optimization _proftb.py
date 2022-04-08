import mala
from mala import printout
from data_repo_path import data_repo_path
data_path = data_repo_path+"Be2/densities_gp/"

from pathlib import Path
Path("./tb_logs/gp").mkdir(parents=True, exist_ok=True)

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


seed_all(5476)

#torch-tb-profiler and tensorboard need to be installed to view the trace files 
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             #schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=2),
             on_trace_ready=torch.profiler.tensorboard_trace_handler('./tb_logs/gp'),
             record_shapes=True,
             profile_memory=True,
             with_stack=True) as prof:
    """
    ex13_gaussian_processes_optimization.py: Shows how Gaussian processes can be 
    used to learn the electronic density with MALA. Backend is GPytorch.
    Here, an optimization is performed in the sense that the model (hyper-)
    parameters are optimized. It is similar to ex04 in that regard.
    """

    params = mala.Parameters()
    params.use_gpu = True

    # Specify the data scaling.
    params.data.input_rescaling_type = "feature-wise-standard"
    params.data.output_rescaling_type = "normal"

    # Specify the used activation function.
    params.model.loss_function_type = "gaussian_likelihood"
    params.model.kernel = "rbf"
    #params.model.kernel = "rbf+linear"

    # Specify the training parameters.
    params.running.max_number_epochs = 20

    # This should be 1, and MALA will set it automatically to, if we don't.
    params.running.mini_batch_size = 40
    params.running.learning_rate = 0.1
    params.running.trainingtype = "Adam"
    params.targets.target_type = "Density"
    #params.debug.grid_dimensions = [10, 10, 1]

    with record_function("data_loader"):
        ####################
        # DATA
        # Add and prepare snapshots for training.
        ####################

        data_handler = mala.DataHandler(params)
        inputs_folder = data_path+"inputs_snap/"
        outputs_folder = data_path+"outputs_density/"
        additional_folder = data_path+"additional_info_qeouts/"
        # Add a snapshot we want to use in to the list.
        data_handler.add_snapshot("snapshot0.in.npy", inputs_folder,
                                "snapshot0.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
        data_handler.add_snapshot("snapshot1.in.npy", inputs_folder,
                                "snapshot1.out.npy", outputs_folder, add_snapshot_as="va", output_units="None")
        data_handler.add_snapshot("snapshot2.in.npy", inputs_folder,
                                "snapshot2.out.npy", outputs_folder, add_snapshot_as="te",
                                output_units="None", calculation_output_file=additional_folder+"snapshot2.out")
        data_handler.prepare_data(transpose_data=True)
        printout("Read data: DONE.")

    with record_function("network_setup"):
        ####################
        # MODEL SETUP
        # Set up the model and trainer we want to use.
        ####################
        model = mala.GaussianProcesses(params, data_handler)
        trainer = mala.Trainer(params, model, data_handler)
        printout("Network setup: DONE.")

    with record_function("optimization"):
        ####################
        # TRAINING
        # Train the network.
        ####################

        printout("Starting training.")
        trainer.train_model()
        printout("Training: DONE.")

    with record_function("tester_setup"):
        ####################
        # TESTING
        # Pass the first test set snapshot (the test snapshot).
        ####################

        tester = mala.Tester(params, model, data_handler)

    with record_function("inference"):    
        actual_density, predicted_density = tester.test_snapshot(0)

    with record_function("target_calculation"):
        # First test snapshot --> 2nd in total
        data_handler.target_calculator.read_additional_calculation_data("qe.out", data_handler.get_snapshot_calculation_output(2))
        actual_number_of_electrons = data_handler.target_calculator.get_number_of_electrons(actual_density)
        predicted_number_of_electrons = data_handler.target_calculator.get_number_of_electrons(predicted_density)
        printout(f"actual_number_of_electrons: {actual_number_of_electrons}, predicted_number_of_electrons: {predicted_number_of_electrons}")
