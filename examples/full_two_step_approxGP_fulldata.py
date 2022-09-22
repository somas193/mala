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
params_gp1.data.input_rescaling_type = "feature-wise-standard"
params_gp1.data.output_rescaling_type = "normal"

# Specify the used activation function.
params_gp1.model.variational_dist_type = "cholesky"
params_gp1.model.variational_strategy_type = "variational_strategy"
params_gp1.model.multivariate_distribution = "normal"
params_gp1.model.loss_function_type = "gaussian_likelihood"
params_gp1.model.max_log_likelihood = "elbo"
params_gp1.model.kernel = "rbf"

# Specify the training parameters.
params_gp1.running.max_number_epochs = 100

# This should be 1, and MALA will set it automatically to, if we don't.
params_gp1.running.mini_batch_size = 1000
params_gp1.running.learning_rate = 0.01
params_gp1.running.trainingtype = "Adam"
params_gp1.targets.target_type = "Density"
#params_gp1.debug.grid_dimensions = [10, 10, 1]

####################
# DATA
# Add and prepare snapshots for training.
####################

data_handler_gp1 = mala.DataHandler(params_gp1)
inputs_folder_gp1 = "/home/rofl/MALA/test-data/Be2/training_data/"
outputs_folder_gp1 = data_path+"outputs_density/"
additional_folder_gp1 = data_path+"additional_info_qeouts/"
# Add a snapshot we want to use in to the list.
data_handler_gp1.add_snapshot("Be_snapshot1.in.npy", inputs_folder_gp1,
                              "snapshot1.out.npy", outputs_folder_gp1, add_snapshot_as="tr", output_units="None")
data_handler_gp1.add_snapshot("Be_snapshot2.in.npy", inputs_folder_gp1,
                              "snapshot2.out.npy", outputs_folder_gp1, add_snapshot_as="va", output_units="None")
data_handler_gp1.add_snapshot("Be_snapshot3.in.npy", inputs_folder_gp1,
                              "snapshot3.out.npy", outputs_folder_gp1, add_snapshot_as="te",
                              output_units="None", calculation_output_file=additional_folder_gp1+"snapshot3.out")
data_handler_gp1.prepare_data()
#print(data_handler.training_data_inputs.size())
printout("Read data: DONE.")
inducing_points = data_handler_gp1.get_inducing_points(2000)
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
printout(f"actual_number_of_electrons: {actual_number_of_electrons}, predicted_number_of_electrons: {predicted_number_of_electrons}")

np.save("/media/rofl/New Volume/TU Dresden/THESIS/actual_electronic_density_f", actual_density)
np.save("/media/rofl/New Volume/TU Dresden/THESIS/predicted_electronic_density_f", predicted_density)

params_gp2 = mala.Parameters()
params_gp2.use_gpu = True
params_gp2.use_multitask_gp = True

# Specify the data scaling.
params_gp2.data.input_rescaling_type = "normal"
params_gp2.data.output_rescaling_type = "feature-wise-standard"

# Specify the used activation function.
params_gp2.model.variational_dist_type = "cholesky"
params_gp2.model.variational_strategy_type = "LMC"
params_gp2.model.loss_function_type = "multitask"
params_gp2.model.multivariate_distribution = "normal"
params_gp2.model.max_log_likelihood = "elbo"
params_gp2.model.kernel = "rbf"
params_gp2.model.no_of_tasks = 2
params_gp2.model.no_of_latents = 2

# Specify the training parameters.
params_gp2.running.max_number_epochs = 100

# This should be 1, and MALA will set it automatically to, if we don't.
params_gp2.running.mini_batch_size = 1000
params_gp2.running.learning_rate = 0.01
params_gp2.running.trainingtype = "Adam"
params_gp2.targets.target_type = "Energy density"

####################
# DATA
# Add and prepare snapshots for training.
####################

data_handler_gp2 = mala.DataHandler(params_gp2)
inputs_folder_gp2 = data_path+"outputs_density/"
outputs_folder_gp2 = "/home/rofl/MALA_fork/"
additional_folder_gp2 = data_path+"additional_info_qeouts/"

# Add a snapshot we want to use in to the list.
data_handler_gp2.add_snapshot("snapshot1.out.npy", inputs_folder_gp2,
                              "Be2_ed_snapshot1.npy", outputs_folder_gp2, add_snapshot_as="tr", output_units="None")
data_handler_gp2.add_snapshot("snapshot2.out.npy", inputs_folder_gp2,
                              "Be2_ed_snapshot2.npy", outputs_folder_gp2, add_snapshot_as="va", output_units="None")
data_handler_gp2.add_snapshot("snapshot3.out.npy", inputs_folder_gp2,
                              "Be2_ed_snapshot3.npy", outputs_folder_gp2, add_snapshot_as="te",
                               output_units="None", calculation_output_file=additional_folder_gp2+"snapshot3.out")
# data_handler_gp.add_snapshot("red_snapshot1.out.npy", outputs_folder_gp,
#                               "red_Be2_ed_snapshot1.npy", outputs_folder_gp, add_snapshot_as="tr", output_units="None")
# data_handler_gp.add_snapshot("red_snapshot2.out.npy", outputs_folder_gp,
#                               "red_Be2_ed_snapshot2.npy", outputs_folder_gp, add_snapshot_as="va", output_units="None")
# data_handler_gp.add_snapshot("red_snapshot3.out.npy", outputs_folder_gp,
#                               "red_Be2_ed_snapshot3.npy", outputs_folder_gp, add_snapshot_as="te",
#                                output_units="None", calculation_output_file=additional_folder+"snapshot3.out")                            
data_handler_gp2.prepare_data()
printout("Read data: DONE.")
inducing_points = data_handler_gp2.get_inducing_points(2000, params_gp2.model.no_of_latents)
#print(inducing_points)

####################
# Model SETUP
# Set up the model and trainer we want to use.
####################
model_gp2 = mala.ApproxGaussianProcesses(params_gp2, inducing_points)
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
actual_ed, predicted_ed_simdata = tester_gp2.test_snapshot(0)
predicted_ed = tester_gp2.predict_from_array(torch.from_numpy(predicted_density).float())

print("\nActual energy density: {}, Predicted energy density: {}".format(actual_ed, predicted_ed))

# for dev in range(num_gpus):
# 	print(torch.cuda.memory_summary(f'cuda:{dev}')) # print the cuda memory usage
	
# # First test snapshot --> 2nd in total
# data_handler_gp1.target_calculator.read_additional_calculation_data("qe.out", data_handler_gp1.get_snapshot_calculation_output(4))
# actual_number_of_electrons = data_handler_gp1.target_calculator.get_number_of_electrons(actual_density)
# predicted_number_of_electrons = data_handler_gp1.target_calculator.get_number_of_electrons(predicted_density)
# printout(f"actual_number_of_electrons: {actual_number_of_electrons}, predicted_number_of_electrons: {predicted_number_of_electrons}")

data_handler_gp2.target_calculator.read_additional_calculation_data("qe.out", data_handler_gp2.get_snapshot_calculation_output(2))
actual_integrated_be = data_handler_gp2.target_calculator.get_integrated_quantities(actual_ed[:,0])
actual_integrated_ec = data_handler_gp2.target_calculator.get_integrated_quantities(actual_ed[:,1])

predicted_integrated_be = data_handler_gp2.target_calculator.get_integrated_quantities(predicted_ed[:,0])
predicted_integrated_ec = data_handler_gp2.target_calculator.get_integrated_quantities(predicted_ed[:,1])
printout(f"actual band energy: {actual_integrated_be}, predicted band energy: {predicted_integrated_be}")
printout(f"actual entropy contribution: {actual_integrated_ec}, predicted entropy contribution: {predicted_integrated_ec}")

integrated_energy = {"actual_be":actual_integrated_be, "actual_ec":actual_integrated_ec,
                     "predicted_be":predicted_integrated_be, "predicted_ec":predicted_integrated_ec}


np.save("/media/rofl/New Volume/TU Dresden/THESIS/actual_energy_density_f", actual_ed)
np.save("/media/rofl/New Volume/TU Dresden/THESIS/predicted_energy_density_f", predicted_ed)
np.save("/media/rofl/New Volume/TU Dresden/THESIS/predicted_energy_density_simdata_f", predicted_ed_simdata)
with open("/media/rofl/New Volume/TU Dresden/THESIS/integrated_energy_f.pickle", 'wb') as f:
    pickle.dump(integrated_energy, f)