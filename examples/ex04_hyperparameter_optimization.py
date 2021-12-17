import mala
from mala import printout
from data_repo_path import get_data_repo_path
data_path = get_data_repo_path()+"Al36/"

"""
ex04_hyperparameter_optimization.py: Shows how a hyperparameter 
optimization can be done using this framework. There are multiple 
hyperparameter optimizers available in this framework. This example focusses
on the most universal one - optuna.  
"""


####################
# PARAMETERS
# All parameters are handled from a central parameters class that
# contains subclasses.
####################
test_parameters = mala.Parameters()
# Currently, the splitting in training, validation and test set are
# done on a "by snapshot" basis. Specify how this is
# done by providing a list containing entries of the form
# "tr", "va" and "te".
test_parameters.data.data_splitting_type = "by_snapshot"

# Specify the data scaling.
test_parameters.data.input_rescaling_type = "feature-wise-standard"
test_parameters.data.output_rescaling_type = "normal"

# Specify the training parameters.
test_parameters.running.max_number_epochs = 20
test_parameters.running.mini_batch_size = 40
test_parameters.running.learning_rate = 0.00001
test_parameters.running.trainingtype = "Adam"
test_parameters.targets.target_type = "Density"

# Specify the number of trials, the hyperparameter optimizer should run
# and the type of hyperparameter.
test_parameters.hyperparameters.n_trials = 20
test_parameters.hyperparameters.hyper_opt_method = "optuna"

####################
# DATA
# Add and prepare snapshots for training.
####################
data_handler = mala.DataHandler(test_parameters)

# Add a snapshot we want to use in to the list.
inputs_folder = data_path+"inputs_snap/"
outputs_folder = data_path+"outputs_density/"
additional_folder = data_path+"additional_info_qeouts/"
data_handler.add_snapshot("snapshot0.in.npy", inputs_folder,
                          "snapshot0.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
data_handler.add_snapshot("snapshot1.in.npy", inputs_folder,
                          "snapshot1.out.npy", outputs_folder, add_snapshot_as="va", output_units="None")
data_handler.add_snapshot("snapshot2.in.npy", inputs_folder,
                          "snapshot2.out.npy", outputs_folder, add_snapshot_as="te",
                          output_units="None", calculation_output_file=additional_folder+"snapshot2.out")
data_handler.prepare_data()
printout("Read data: DONE.")

####################
# HYPERPARAMETER OPTIMIZATION
# In order to perform a hyperparameter optimization,
# one has to simply create a hyperparameter optimizer
# and let it perform a "study".
# Before such a study can be done, one has to add all the parameters
# of interest.
####################

test_hp_optimizer = mala.HyperOptInterface(test_parameters, data_handler)

# Learning rate will be optimized.
test_hp_optimizer.add_hyperparameter("float", "learning_rate",
                                     0.0000001, 0.01)

# Number of neurons per layer will be optimized.
test_hp_optimizer.add_hyperparameter("int", "ff_neurons_layer_00", 10, 100)
test_hp_optimizer.add_hyperparameter("int", "ff_neurons_layer_01", 10, 100)

# Choices for activation function at each layer will be optimized.
test_hp_optimizer.add_hyperparameter("categorical", "layer_activation_00",
                                     choices=["ReLU", "Sigmoid"])
test_hp_optimizer.add_hyperparameter("categorical", "layer_activation_01",
                                     choices=["ReLU", "Sigmoid"])
test_hp_optimizer.add_hyperparameter("categorical", "layer_activation_02",
                                     choices=["ReLU", "Sigmoid"])

# Perform hyperparameter optimization.
printout("Starting Hyperparameter optimization.")
test_hp_optimizer.perform_study()
test_hp_optimizer.set_optimal_parameters()
printout("Hyperparameter optimization: DONE.")

####################
# TRAINING
# Train with these new parameters.
####################

test_network = mala.Network(test_parameters)
test_trainer = mala.Trainer(test_parameters, test_network, data_handler)
printout("Network setup: DONE.")
test_trainer.train_model()
printout("Training: DONE.")

####################
# RESULTS.
# Print the used parameters and check whether the loss decreased enough.
####################

printout("Parameters used for this experiment:")
test_parameters.show()
