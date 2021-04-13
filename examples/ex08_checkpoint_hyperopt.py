import mala
from mala import printout
from data_repo_path import get_data_repo_path
data_path = get_data_repo_path()+"Al36/"

"""
ex08_checkpoint_hyperopt.py: Shows how a hyperparameter optimization run can 
be paused and resumed.
"""


def run_example08():
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
    test_parameters.data.data_splitting_snapshots = ["tr", "va", "te"]

    # Specify the data scaling.
    test_parameters.data.input_rescaling_type = "feature-wise-standard"
    test_parameters.data.output_rescaling_type = "normal"

    # Specify the training parameters.
    test_parameters.running.max_number_epochs = 10
    test_parameters.running.mini_batch_size = 40
    test_parameters.running.learning_rate = 0.00001
    test_parameters.running.trainingtype = "Adam"

    # Specify the number of trials, the hyperparameter optimizer should run
    # and the type of hyperparameter.
    # The study will be 9 trials long, with checkpoints after the 5th
    # trial. This will result in the first checkpoint still having 4
    # trials left.
    test_parameters.hyperparameters.n_trials = 9
    test_parameters.hyperparameters.hyper_opt_method = "optuna"
    test_parameters.hyperparameters.checkpoints_each_trial = 5
    test_parameters.hyperparameters.checkpoint_name = "ex08"

    ####################
    # DATA
    # Add and prepare snapshots for training.
    ####################
    data_handler = mala.DataHandler(test_parameters)

    # Add all the snapshots we want to use in to the list.
    data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path,
                              "Al_debug_2k_nr0.out.npy", data_path,
                              output_units="1/Ry")
    data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path,
                              "Al_debug_2k_nr1.out.npy", data_path,
                              output_units="1/Ry")
    data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                              "Al_debug_2k_nr2.out.npy", data_path,
                              output_units="1/Ry")
    data_handler.prepare_data()
    printout("Read data: DONE.")

    ####################
    # HYPERPARAMETER OPTIMIZATION
    # In order to perform a hyperparameter optimization,
    # one has to simply create a hyperparameter optimizer
    # and let it perform a "study".
    # Before such a study can be done, one has to add all the parameters
    # of interest.
    # After first performing the study, we load it from the last checkpoint
    # and run the remaining trials.
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

    loaded_params, new_datahandler, new_hyperopt = \
        mala.HyperOptOptuna.resume_checkpoint("ex08")
    new_hyperopt.perform_study()

    ####################
    # RESULTS.
    # Print the used parameters.
    ####################

    printout("Parameters used for this experiment:")
    test_parameters.show()
    return True


if __name__ == "__main__":
    if run_example08():
        printout("Successfully ran ex08_checkpoint_hyperopt.")
    else:
        raise Exception("Ran ex08_checkpoint_hyperopt but something was off."
                        " If you haven't changed any parameters in "
                        "the example, there might be a problem with your"
                        " installation.")
