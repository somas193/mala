import mala
import torch
import numpy as np
from mala import printout
from data_repo_path import data_repo_path
data_path = data_repo_path+"Be2/densities_gp/"

params_gp2 = mala.Parameters()
params_gp2.use_gpu = False
params_gp2.use_multitask_gp = True

# Specify the data scaling.
params_gp2.data.input_rescaling_type = "feature-wise-standard"
#params_gp2.data.output_rescaling_type = "feature-wise-standard"

# Specify the used activation function.
params_gp2.model.loss_function_type = "multitask"
params_gp2.model.multivariate_distribution = "multitask-normal"
params_gp2.model.kernel = "rbf"
params_gp2.model.no_of_tasks = 2
params_gp2.model.rank = 1

# Specify the training parameters.
params_gp2.running.max_number_epochs = 1

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
outputs_folder_gp2 = "/home/kulkar74/MALA_fork/"
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
print(type(actual_ed))
print(type(predicted_ed))
print(actual_ed.shape)
print(predicted_ed.shape)

#print("\nActual energy density: {}, Predicted energy density: {}".format(actual_ed, predicted_ed))

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

# #Pseudopotential path
# psp_path = "/home/rofl/MALA/test-data/Be2/Be.pbe-n-rrkjus_psl.1.0.0.UPF"

# #LDOS path
# ldos_data_path = "/home/rofl/MALA/test-data/Be2/training_data/Be_snapshot3.out.npy"

# ####################
# # PARAMETERS
# # All parameters are handled from a central parameters class that
# # contains subclasses.
# ####################
# test_parameters = mala.Parameters()

# # Specify the correct LDOS parameters.
# test_parameters.targets.target_type = "LDOS"
# test_parameters.targets.ldos_gridsize = 11
# test_parameters.targets.ldos_gridspacing_ev = 2.5
# test_parameters.targets.ldos_gridoffset_ev = -5
# # To perform a total energy calculation one also needs to provide
# # a pseudopotential(path).
# test_parameters.targets.pseudopotential_path = psp_path

# ####################
# # TARGETS
# # Create a target calculator to postprocess data.
# # Use this calculator to perform various operations.
# ####################

# ldos = mala.TargetInterface(test_parameters)

# # Read additional information about the calculation.
# # By doing this, the calculator is able to know e.g. the temperature
# # at which the calculation took place or the lattice constant used.
# ldos.read_additional_calculation_data("qe.out",
#                                       "/home/rofl/MALA/test-data/Be2/training_data/Be_snapshot3.out")

# ldos_data = np.load(ldos_data_path)
# ldos_data = ldos_data[:12, :12, :18]
# print(ldos_data.shape)
# print(ldos_data[0,2,3,:])

# Get quantities of interest.
# For better values in the post processing, it is recommended to
# calculate the "self-consistent Fermi energy", i.e. the Fermi energy
# at which the (L)DOS reproduces the exact number of electrons.
# This Fermi energy usually differs from the one outputted by the
# QuantumEspresso calculation, due to numerical reasons. The difference
# is usually very small.
#self_consistent_fermi_energy = ldos.\
#    get_self_consistent_fermi_energy_ev(ldos_data)
#Compute band energy and entropy contribution
# energy_density = ldos.get_energy_density(ldos_data, self_consistent_fermi_energy)
# ed_calculator = mala.targets.EnergyDensity(test_parameters)
# ed_calculator.read_additional_calculation_data('qe.out', "/home/rofl/MALA/test-data/Be2/training_data/Be_snapshot3.out")
# #Band energy and entropy calculation through integration
# band_energy_integrated = ed_calculator.get_integrated_quantities(energy_density[:,0])
# entropy_integrated = ed_calculator.get_integrated_quantities(energy_density[:,1])

# band_energy_direct = ldos.get_band_energy(ldos_data, self_consistent_fermi_energy)
# entropy_direct = ldos.get_entropy_contribution(ldos_data, self_consistent_fermi_energy)

# printout(f"actual be: {actual_integrated_be}, predicted be: {predicted_integrated_be}, direct be: {band_energy_direct} ")
# printout(f"actual ec: {actual_integrated_ec}, predicted ec: {predicted_integrated_ec}, direct ec: : {entropy_direct}")