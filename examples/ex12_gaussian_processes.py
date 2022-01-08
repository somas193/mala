import torch
import mala
from mala import printout
from data_repo_path import data_repo_path
data_path = data_repo_path+"Be2/densities_gp/"

"""
ex12_gassian_processes.py: Shows how Gaussian processes can be used
to learn the electronic density with MALA. Backend is GPytorch.
This is a "Single Shot" Gaussian process, meaning we do not optimize the hyper-
parameters (it is the equivalent to ex01 in that regard.)
"""

params = mala.Parameters()
params.use_gpu = True
params.manual_seed = 14012022

# Specify the data scaling.
params.data.input_rescaling_type = "feature-wise-standard"
params.data.output_rescaling_type = "normal"

# Specify the used activation function.
params.model.loss_function_type = "gaussian_likelihood"

# Specify the training parameters.
params.running.max_number_epochs = 20

# This should be 1, and MALA will set it automatically to, if we don't.
params.running.mini_batch_size = 40
params.running.learning_rate = 0.1
params.running.trainingtype = "Adam"
params.targets.target_type = "Density"
#params.debug.grid_dimensions = [10, 10, 1]
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
                          "snapshot1.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
data_handler.add_snapshot("snapshot2.in.npy", inputs_folder,
                          "snapshot2.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
                          
data_handler.add_snapshot("snapshot3.in.npy", inputs_folder,
                          "snapshot3.out.npy", outputs_folder, add_snapshot_as="va", output_units="None")
data_handler.add_snapshot("snapshot4.in.npy", inputs_folder,
                          "snapshot4.out.npy", outputs_folder, add_snapshot_as="te",
                          output_units="None", calculation_output_file=additional_folder+"snapshot4.out")
data_handler.prepare_data(transpose_data=True)
printout("Read data: DONE.")

####################
# MODEL SETUP
# Set up the model.
# Gaussian Processes do not have to be trained in order
# to captue the trainint data.
####################
params.model.kernel = "rbf+linear"
model = mala.GaussianProcesses(params, data_handler)

####################
# TESTING
# Pass the first test set snapshot (the test snapshot).
####################

tester = mala.Tester(params, model, data_handler)
actual_density, predicted_density = tester.test_snapshot(0)
print(torch.cuda.memory_summary()) # print the cuda memory usage
# First test snapshot --> 2nd in total
data_handler.target_calculator.read_additional_calculation_data("qe.out", data_handler.get_snapshot_calculation_output(4))
actual_number_of_electrons = data_handler.target_calculator.get_number_of_electrons(actual_density)
predicted_number_of_electrons = data_handler.target_calculator.get_number_of_electrons(predicted_density)
printout(f"actual_number_of_electrons: {actual_number_of_electrons}, predicted_number_of_electrons: {predicted_number_of_electrons}")
