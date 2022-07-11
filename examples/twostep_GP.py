import mala
import torch
import numpy as np
from mala import printout
from data_repo_path import data_repo_path
data_path = data_repo_path+"Be2/densities_gp/"

"""
ex13_gaussian_processes_optimization.py: Shows how Gaussian processes can be 
used to learn the electronic density with MALA. Backend is GPytorch.
Here, an optimization is performed in the sense that the model_gp1 (hyper-)
parameters are optimized. It is similar to ex04 in that regard.
"""

params_gp1 = mala.Parameters()
params_gp1.use_gpu = False

# Specify the data scaling.
params_gp1.data.input_rescaling_type = "feature-wise-standard"
params_gp1.data.output_rescaling_type = "normal"

# Specify the used activation function.
params_gp1.model.loss_function_type = "gaussian_likelihood"
params_gp1.model.kernel = "rbf"
#params_gp1.model_gp1.kernel = "rbf+linear"

# Specify the training parameters.
params_gp1.running.max_number_epochs = 18

# This should be 1, and MALA will set it automatically to, if we don't.
params_gp1.running.mini_batch_size = 40
params_gp1.running.learning_rate = 0.1
params_gp1.running.trainingtype = "Adam"
params_gp1.targets.target_type = "Density"
#params_gp1.debug.grid_dimensions = [10, 10, 1]
####################
# DATA
# Add and prepare snapshots for training.
####################

data_handler_gp1 = mala.DataHandler(params_gp1)
inputs_folder_gp1 = data_path+"inputs_snap/"
outputs_folder_gp1 = data_path+"outputs_density/"
additional_folder = data_path+"additional_info_qeouts/"
# Add a snapshot we want to use in to the list.
# data_handler_gp1.add_snapshot("snapshot0.in.npy", inputs_folder,
#                           "snapshot0.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
data_handler_gp1.add_snapshot("snapshot1.in.npy", inputs_folder_gp1,
                              "snapshot1.out.npy", outputs_folder_gp1, add_snapshot_as="tr", output_units="None")
data_handler_gp1.add_snapshot("snapshot2.in.npy", inputs_folder_gp1,
                              "snapshot2.out.npy", outputs_folder_gp1, add_snapshot_as="va", output_units="None")
data_handler_gp1.add_snapshot("snapshot3.in.npy", inputs_folder_gp1,
                              "snapshot3.out.npy", outputs_folder_gp1, add_snapshot_as="te",
                               output_units="None", calculation_output_file=additional_folder+"snapshot3.out")
data_handler_gp1.prepare_data(transpose_data=True)
printout("Read data: DONE.")

####################
# Model SETUP
# Set up the model and trainer we want to use.
####################
model_gp1 = mala.GaussianProcesses(params_gp1, data_handler_gp1)
trainer_gp1 = mala.Trainer(params_gp1, model_gp1, data_handler_gp1)
printout("Network setup: DONE.")

####################
# TRAINING
# Train the network.
####################

printout("Starting training.")
trainer_gp1.train_model()
printout("Training: DONE.")


####################
# TESTING
# Pass the first test set snapshot (the test snapshot).
####################

tester_gp1 = mala.Tester(params_gp1, model_gp1, data_handler_gp1)
actual_density, predicted_density = tester_gp1.test_snapshot(0)
print(type(actual_density))
print(type(predicted_density))

print("\nActual density: {}, Predicted density: {}".format(actual_density, predicted_density))

params_gp2 = mala.Parameters()
params_gp2.use_gpu = False
params_gp2.use_multitask_gp = True

# Specify the data scaling.
params_gp2.data.input_rescaling_type = "feature-wise-standard"
params_gp2.data.output_rescaling_type = "multitask-normal"

# Specify the used activation function.
params_gp2.model.loss_function_type = "multitask"
params_gp2.model.kernel = "rbf"
params_gp2.model.no_of_tasks = 2
params_gp2.model.rank = 1

# Specify the training parameters.
params_gp2.running.max_number_epochs = 18

# This should be 1, and MALA will set it automatically to, if we don't.
params_gp2.running.mini_batch_size = 1000
params_gp2.running.learning_rate = 0.1
params_gp2.running.trainingtype = "Adam"
params_gp2.targets.target_type = "Energy density"

####################
# DATA
# Add and prepare snapshots for training.
####################

data_handler_gp2 = mala.DataHandler(params_gp2)
inputs_folder_gp2 = data_path+"outputs_density/"
outputs_folder_gp2 = "/home/rofl/MALA_fork/"
additional_folder = data_path+"additional_info_qeouts/"

# Add a snapshot we want to use in to the list.
data_handler_gp2.add_snapshot("snapshot1.out.npy", inputs_folder_gp2,
                              "Be2_ed_snapshot1.npy", outputs_folder_gp2, add_snapshot_as="tr", output_units="None")
data_handler_gp2.add_snapshot("snapshot2.out.npy", inputs_folder_gp2,
                              "Be2_ed_snapshot2.npy", outputs_folder_gp2, add_snapshot_as="va", output_units="None")
data_handler_gp2.add_snapshot("snapshot3.out.npy", inputs_folder_gp2,
                              "Be2_ed_snapshot3.npy", outputs_folder_gp2, add_snapshot_as="te",
                               output_units="None", calculation_output_file=additional_folder+"snapshot3.out")
in_data = np.load(inputs_folder_gp2 + "snapshot1.out.npy")
out_data = np.load(outputs_folder_gp2 + "Be2_ed_snapshot1.npy")
print(in_data.shape)
print(out_data.shape)                             
data_handler_gp2.prepare_data(transpose_data=True)
printout("Read data: DONE.")

####################
# Model SETUP
# Set up the model and trainer we want to use.
####################
model_gp2 = mala.GaussianProcesses(params_gp2, data_handler_gp2)
trainer_gp2 = mala.Trainer(params_gp2, model_gp2, data_handler_gp2)
printout("Network setup: DONE.")

####################
# TRAINING
# Train the network.
####################

printout("Starting training.")
trainer_gp2.train_model()
printout("Training: DONE.")


####################
# TESTING
# Pass the first test set snapshot (the test snapshot).
####################

tester_gp2 = mala.Tester(params_gp2, model_gp2, data_handler_gp2)
actual_ed, predicted_ed = tester_gp2.test_snapshot(0)

print("\nActual energy density: {}, Predicted energy density: {}".format(actual_ed, predicted_ed))

# for dev in range(num_gpus):
# 	print(torch.cuda.memory_summary(f'cuda:{dev}')) # print the cuda memory usage
	
# # First test snapshot --> 2nd in total
# data_handler_gp1.target_calculator.read_additional_calculation_data("qe.out", data_handler_gp1.get_snapshot_calculation_output(4))
# actual_number_of_electrons = data_handler_gp1.target_calculator.get_number_of_electrons(actual_density)
# predicted_number_of_electrons = data_handler_gp1.target_calculator.get_number_of_electrons(predicted_density)
# printout(f"actual_number_of_electrons: {actual_number_of_electrons}, predicted_number_of_electrons: {predicted_number_of_electrons}")