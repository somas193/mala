import mala
import numpy as np
import gpytorch
import torch
from mala import printout
from data_repo_path import data_repo_path
data_path = data_repo_path+"Be2/densities_gp/"

"""
ex13_gaussian_processes_optimization.py: Shows how Gaussian processes can be 
used to learn the electronic density with MALA. Backend is GPytorch.
Here, an optimization is performed in the sense that the model (hyper-)
parameters are optimized. It is similar to ex04 in that regard.
"""
#gpytorch.settings.lazily_evaluate_kernels(True)
params = mala.Parameters()
params.use_gpu = True

# Specify the data scaling.
params.data.input_rescaling_type = "feature-wise-standard"
params.data.output_rescaling_type = "normal"

# Specify the used activation function.
params.model.loss_function_type = "gaussian_likelihood"
params.model.kernel = "periodic"
params.model.gp_mean = "zero"
#params.model.kernel = "rbf+linear"

# Specify the training parameters.
params.running.max_number_epochs = 50
# This should be 1, and MALA will set it automatically to, if we don't.
params.running.mini_batch_size = 40
params.running.learning_rate = 0.1
params.running.trainingtype = "Adam"
params.targets.target_type = "Density"
params.data.descriptors_contain_xyz = False
#params.debug.grid_dimensions = [10, 10, 1]
####################
# DATA
# Add and prepare snapshots for training.
####################

# data_handler = mala.DataHandler(params)
# inputs_folder_gp1 = "/home/kulkar74/MALA/test-data/Be2/training_data/"
# outputs_folder_gp1 = data_path+"outputs_density/"
# additional_folder_gp1 = data_path+"additional_info_qeouts/"

# inputs_folder = data_path+"inputs_snap/"
# outputs_folder = data_path+"outputs_density/"
# additional_folder = data_path+"additional_info_qeouts/"
# # Add a snapshot we want to use in to the list.
# data_handler.add_snapshot("snapshot1.in.npy", inputs_folder,
#                           "snapshot1.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
# data_handler.add_snapshot("snapshot2.in.npy", inputs_folder,
#                           "snapshot2.out.npy", outputs_folder, add_snapshot_as="va", output_units="None")
# data_handler.add_snapshot("snapshot3.in.npy", inputs_folder,
#                           "snapshot3.out.npy", outputs_folder, add_snapshot_as="te",
#                           output_units="None", calculation_output_file=additional_folder+"snapshot3.out")
# data_handler.prepare_data(transpose_data=True)

# inputs_folder = data_path+"inputs_snap/"
# outputs_folder = data_path+"outputs_density/"
# additional_folder = data_path+"additional_info_qeouts/"
# # Add a snapshot we want to use in to the list.

# data_handler.add_snapshot("optimized_snapshot1.in.npy", inputs_folder,
#                           "snapshot1.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
# data_handler.add_snapshot("optimized_snapshot2.in.npy", inputs_folder,
#                           "snapshot2.out.npy", outputs_folder, add_snapshot_as="va", output_units="None")
# data_handler.add_snapshot("optimized_snapshot3.in.npy", inputs_folder,
#                           "snapshot3.out.npy", outputs_folder, add_snapshot_as="te",
#                           output_units="None", calculation_output_file=additional_folder+"snapshot3.out")
# data_handler.prepare_data(transpose_data=True)

data_handler = mala.DataHandler(params)
inputs_folder = data_path+"inputs_gaussians/"
outputs_folder = data_path+"outputs_density/"
additional_folder = data_path+"additional_info_qeouts/"
# Add a snapshot we want to use in to the list.

data_handler.add_snapshot("gaussians1.npy", inputs_folder,
                          "snapshot1.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
data_handler.add_snapshot("gaussians2.npy", inputs_folder,
                          "snapshot2.out.npy", outputs_folder, add_snapshot_as="va", output_units="None")
data_handler.add_snapshot("gaussians3.npy", inputs_folder,
                          "snapshot3.out.npy", outputs_folder, add_snapshot_as="te",
                          output_units="None", calculation_output_file=additional_folder+"snapshot3.out")
data_handler.prepare_data(transpose_data=True)
printout("Read data: DONE.")

####################
# MODEL SETUP
# Set up the model and trainer we want to use.
####################
model = mala.GaussianProcesses(params, data_handler)
trainer = mala.Trainer(params, model, data_handler)
printout("Network setup: DONE.")

####################
# TRAINING
# Train the network.
####################

printout("Starting training.")
trainer.train_model()
printout("Training: DONE.")

####################
# TESTING
# Pass the first test set snapshot (the test snapshot).
####################

tester = mala.Tester(params, model, data_handler)
actual_density, predicted_density = tester.test_snapshot(0)
# First test snapshot --> 2nd in total
data_handler.target_calculator.read_additional_calculation_data("qe.out", data_handler.get_snapshot_calculation_output(2))
actual_number_of_electrons = data_handler.target_calculator.get_number_of_electrons(actual_density)
predicted_number_of_electrons = data_handler.target_calculator.get_number_of_electrons(predicted_density)
print("\nActual electronic density: {}, Predicted electronic density: {}".format(actual_density, predicted_density))
printout(f"actual_number_of_electrons: {actual_number_of_electrons}, predicted_number_of_electrons: {predicted_number_of_electrons}")

np.save("/home/kulkar74/actual_electronic_density_f", actual_density)
np.save("/home/kulkar74/predicted_electronic_density_f", predicted_density)
