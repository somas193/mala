import torch
import mala
import numpy as np
import pickle
from mala import printout
from data_repo_path import data_repo_path
data_path = data_repo_path+"Be2/densities_gp/"

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

test_parameters_nn1 = mala.Parameters()
test_parameters_nn1.use_gpu = True
#test_parameters_nn1.manual_seed = 14012022
# Currently, the splitting in training, validation and test set are
# done on a "by snapshot" basis. Specify how this is
# done by providing a list containing entries of the form
# "tr", "va" and "te".
test_parameters_nn1.data.data_splitting_type = "by_snapshot"

# Specify the data scaling.
test_parameters_nn1.data.input_rescaling_type = "feature-wise-normal"
test_parameters_nn1.data.output_rescaling_type = "normal"

# Specify the used activation function.
test_parameters_nn1.model.layer_activations = ["LeakyReLU"]

# Specify the training parameters.
test_parameters_nn1.running.max_number_epochs = 100
test_parameters_nn1.running.mini_batch_size = 2000
test_parameters_nn1.running.learning_rate = 0.005
test_parameters_nn1.running.trainingtype = "Adam"
test_parameters_nn1.targets.target_type = "Density"

####################
# DATA
# Add and prepare snapshots for training.
####################

data_handler_nn1 = mala.DataHandler(test_parameters_nn1)

# Add a snapshot we want to use in to the list.
inputs_folder_nn1 = "/home/rofl/MALA/test-data/Be2/training_data/"
outputs_folder_nn1 = data_path+"outputs_density/"
additional_folder_nn1 = data_path+"additional_info_qeouts/"

data_handler_nn1.add_snapshot("Be_snapshot1.in.npy", inputs_folder_nn1,
                              "snapshot1.out.npy", outputs_folder_nn1, add_snapshot_as="tr", output_units="None")
data_handler_nn1.add_snapshot("Be_snapshot2.in.npy", inputs_folder_nn1,
                              "snapshot2.out.npy", outputs_folder_nn1, add_snapshot_as="va", output_units="None")
data_handler_nn1.add_snapshot("Be_snapshot3.in.npy", inputs_folder_nn1,
                              "snapshot3.out.npy", outputs_folder_nn1, add_snapshot_as="te",
                               output_units="None", calculation_output_file=additional_folder_nn1+"snapshot3.out")
data_handler_nn1.prepare_data()
printout("Read data: DONE.")

####################
# NETWORK SETUP
# Set up the models and trainer we want to use.
# The layer sizes can be specified before reading data,
# but it is safer this way.
####################

test_parameters_nn1.model.layer_sizes = [data_handler_nn1.get_input_dimension(),
                                         200,
                                         data_handler_nn1.get_output_dimension()]

# Setup models and trainer.
test_network_nn1 = mala.Network(test_parameters_nn1)
test_trainer_nn1 = mala.Trainer(test_parameters_nn1, test_network_nn1, data_handler_nn1)
printout("Network setup: DONE.")

####################
# TRAINING
# Train the models.
####################

printout("Starting training.")
test_trainer_nn1.train_model()
printout("Training: DONE.")

####################
# RESULTS.
# Print the used parameters and check whether the loss decreased enough.
####################

printout("Parameters used for this experiment:")
test_parameters_nn1.show()

####################
# TESTING
# Pass the first test set snapshot (the test snapshot).
####################
test_parameters_nn1.running.mini_batch_size = 8000 # to forward the entire snap
tester_nn1 = mala.Tester(test_parameters_nn1, test_network_nn1, data_handler_nn1)
actual_density, predicted_density = tester_nn1.test_snapshot(0)
#print(torch.cuda.memory_summary()) # print the cuda memory usage
# First test snapshot --> 2nd in total
data_handler_nn1.target_calculator.read_additional_calculation_data("qe.out", data_handler_nn1.get_snapshot_calculation_output(2))
actual_number_of_electrons = data_handler_nn1.target_calculator.get_number_of_electrons(actual_density)
predicted_number_of_electrons = data_handler_nn1.target_calculator.get_number_of_electrons(predicted_density)
print("\nActual electronic density: {}, Predicted electronic density: {}".format(actual_density, predicted_density))
printout(f"actual_number_of_electrons: {actual_number_of_electrons}, predicted_number_of_electrons: {predicted_number_of_electrons}")

np.save("/media/rofl/New Volume/TU Dresden/THESIS/actual_electronic_density_nn", actual_density)
np.save("/media/rofl/New Volume/TU Dresden/THESIS/predicted_electronic_density_nn", predicted_density)

####################
# PARAMETERS
# All parameters are handled from a central parameters class that
# contains subclasses.
####################

test_parameters_nn2 = mala.Parameters()
test_parameters_nn2.use_gpu = True
#test_parameters_nn2.manual_seed = 14012022
# Currently, the splitting in training, validation and test set are
# done on a "by snapshot" basis. Specify how this is
# done by providing a list containing entries of the form
# "tr", "va" and "te".
test_parameters_nn2.data.data_splitting_type = "by_snapshot"

# Specify the data scaling.
test_parameters_nn2.data.input_rescaling_type = "normal"
test_parameters_nn2.data.output_rescaling_type = "feature-wise-normal"

# Specify the used activation function.
test_parameters_nn2.model.layer_activations = ["LeakyReLU"]

# Specify the training parameters.
test_parameters_nn2.running.max_number_epochs = 100
test_parameters_nn2.running.mini_batch_size = 1000
test_parameters_nn2.running.learning_rate = 0.0001
test_parameters_nn2.running.trainingtype = "Adam"
test_parameters_nn2.targets.target_type = "Energy density"

####################
# DATA
# Add and prepare snapshots for training.
####################

data_handler_nn2 = mala.DataHandler(test_parameters_nn2)

# Add a snapshot we want to use in to the list.
inputs_folder_nn2 = data_path+"outputs_density/"
outputs_folder_nn2 = "/home/rofl/MALA_fork/"
additional_folder_nn2 = data_path+"additional_info_qeouts/"

data_handler_nn2.add_snapshot("snapshot1.out.npy", inputs_folder_nn2,
                              "Be2_ed_snapshot1.npy", outputs_folder_nn2, add_snapshot_as="tr", output_units="None")
data_handler_nn2.add_snapshot("snapshot2.out.npy", inputs_folder_nn2,
                              "Be2_ed_snapshot2.npy", outputs_folder_nn2, add_snapshot_as="va", output_units="None")
data_handler_nn2.add_snapshot("snapshot3.out.npy", inputs_folder_nn2,
                              "Be2_ed_snapshot3.npy", outputs_folder_nn2, add_snapshot_as="te",
                               output_units="None", calculation_output_file=additional_folder_nn2+"snapshot3.out")
data_handler_nn2.prepare_data()
printout("Read data: DONE.")

####################
# NETWORK SETUP
# Set up the models and trainer we want to use.
# The layer sizes can be specified before reading data,
# but it is safer this way.
####################

test_parameters_nn2.model.layer_sizes = [data_handler_nn2.get_input_dimension(),
                                         500,
                                         data_handler_nn2.get_output_dimension()]

# Setup models and trainer.
test_network_nn2 = mala.Network(test_parameters_nn2)
test_trainer_nn2 = mala.Trainer(test_parameters_nn2, test_network_nn2, data_handler_nn2)
printout("Network setup: DONE.")

####################
# TRAINING
# Train the models.
####################

printout("Starting training.")
test_trainer_nn2.train_model()
printout("Training: DONE.")

####################
# RESULTS.
# Print the used parameters and check whether the loss decreased enough.
####################

printout("Parameters used for this experiment:")
test_parameters_nn2.show()

###################
# TESTING
# Pass the first test set snapshot (the test snapshot).
####################
test_parameters_nn2.running.mini_batch_size = 8000 # to forward the entire snap
tester_nn2 = mala.Tester(test_parameters_nn2, test_network_nn2, data_handler_nn2)
actual_ed, predicted_ed_simdata = tester_nn2.test_snapshot(0)
predicted_ed = tester_nn2.predict_from_array(torch.from_numpy(predicted_density).float())
print("\nActual energy density: {}, Predicted energy density: {}".format(actual_ed, predicted_ed))

data_handler_nn2.target_calculator.read_additional_calculation_data("qe.out", data_handler_nn2.get_snapshot_calculation_output(2))
actual_integrated_be = data_handler_nn2.target_calculator.get_integrated_quantities(actual_ed[:,0])
actual_integrated_ec = data_handler_nn2.target_calculator.get_integrated_quantities(actual_ed[:,1])

predicted_integrated_be = data_handler_nn2.target_calculator.get_integrated_quantities(predicted_ed[:,0])
predicted_integrated_ec = data_handler_nn2.target_calculator.get_integrated_quantities(predicted_ed[:,1])
printout(f"actual band energy: {actual_integrated_be}, predicted band energy: {predicted_integrated_be}")
printout(f"actual entropy contribution: {actual_integrated_ec}, predicted entropy contribution: {predicted_integrated_ec}")

integrated_energy = {"actual_be":actual_integrated_be, "actual_ec":actual_integrated_ec,
                     "predicted_be":predicted_integrated_be, "predicted_ec":predicted_integrated_ec}


np.save("/media/rofl/New Volume/TU Dresden/THESIS/actual_energy_density_nn", actual_ed)
np.save("/media/rofl/New Volume/TU Dresden/THESIS/predicted_energy_density_nn", predicted_ed)
np.save("/media/rofl/New Volume/TU Dresden/THESIS/predicted_energy_density_simdata_nn", predicted_ed_simdata)
with open("/media/rofl/New Volume/TU Dresden/THESIS/integrated_energy_nn.pickle", 'wb') as f:
    pickle.dump(integrated_energy, f)
