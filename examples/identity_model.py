import mala
import scipy
import torch
import numpy as np
import pickle
from mala import printout
from data_repo_path import data_repo_path
data_path = data_repo_path+"Be2/densities_gp/"

"""
full_two_step_approxGP.py: Shows how approximate Gaussian processes 
can be used to learn the electronic density and energy density 
(band energy + entopy contribution) with MALA. During inference,
the predicted electronic density is used to predict the energy density.
Backend is GPytorch. Here, an optimization is performed in the sense 
that the model (hyper-) parameters are optimized. It is similar to 
ex04 in that regard.
"""

params_gp1 = mala.Parameters()
params_gp1.use_gpu = True

# Specify the data scaling.
params_gp1.data.input_rescaling_type = "standard"
params_gp1.data.output_rescaling_type = "normal"

# Specify the used activation function.
params_gp1.model.variational_dist_type = "cholesky"
params_gp1.model.variational_strategy_type = "variational_strategy"
params_gp1.model.multivariate_distribution = "normal"
params_gp1.model.loss_function_type = "gaussian_likelihood"
params_gp1.model.max_log_likelihood = "elbo"
params_gp1.model.kernel = "rbf"
params_gp1.model.gp_mean = "constant"

# Specify the training parameters.
params_gp1.running.max_number_epochs = 50

# This should be 1, and MALA will set it automatically to, if we don't.
params_gp1.running.mini_batch_size = 500
params_gp1.running.learning_rate = 0.5
params_gp1.running.trainingtype = "Adam"
params_gp1.targets.target_type = "Density"
params_gp1.data.descriptors_contain_xyz = False
#params_gp1.debug.grid_dimensions = [10, 10, 1]

####################
# DATA
# Add and prepare snapshots for training.
####################

data_handler_gp1 = mala.DataHandler(params_gp1)
inputs_folder_gp1 = data_path+"outputs_density/"
outputs_folder_gp1 = data_path+"outputs_density/"
additional_folder_gp1 = data_path+"additional_info_qeouts/"
# Add a snapshot we want to use in to the list.
data_handler_gp1.add_snapshot("snapshot1.out.npy", inputs_folder_gp1,
                              "snapshot1.out.npy", outputs_folder_gp1, add_snapshot_as="tr", output_units="None")
data_handler_gp1.add_snapshot("snapshot2.out.npy", inputs_folder_gp1,
                              "snapshot2.out.npy", outputs_folder_gp1, add_snapshot_as="va", output_units="None")
data_handler_gp1.add_snapshot("snapshot3.out.npy", inputs_folder_gp1,
                              "snapshot3.out.npy", outputs_folder_gp1, add_snapshot_as="te",
                              output_units="None", calculation_output_file=additional_folder_gp1+"snapshot3.out")
data_handler_gp1.prepare_data()
#print(data_handler.training_data_inputs.size())
printout("Read data: DONE.")
inducing_points = data_handler_gp1.get_inducing_points(1000)
#print(inducing_points)

####################
# MODEL SETUP
# Set up the model and trainer we want to use.
####################
model_gp1 = mala.ApproxGaussianProcesses(params_gp1, inducing_points)
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
density_similarity = scipy.spatial.distance.cosine(actual_density, predicted_density)
print(f'\nCosine distance between actual and predicted density: {density_similarity}\n')
# First test snapshot --> 2nd in total
data_handler_gp1.target_calculator.read_additional_calculation_data("qe.out", data_handler_gp1.get_snapshot_calculation_output(2))
actual_number_of_electrons = data_handler_gp1.target_calculator.get_number_of_electrons(actual_density)
predicted_number_of_electrons = data_handler_gp1.target_calculator.get_number_of_electrons(predicted_density)
print("\nActual electronic density: {}, Predicted electronic density: {}".format(actual_density, predicted_density))
printout(f"actual_number_of_electrons: {actual_number_of_electrons}, predicted_number_of_electrons: {predicted_number_of_electrons}")

np.save("/media/rofl/New Volume/TU Dresden/THESIS/actual_electronic_density_f", actual_density)
np.save("/media/rofl/New Volume/TU Dresden/THESIS/predicted_electronic_density_f", predicted_density)