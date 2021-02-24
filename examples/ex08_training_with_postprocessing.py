import fesl
from fesl import printout
import matplotlib.pyplot as plt
import numpy as np
from data_repo_path import get_data_repo_path
data_path = get_data_repo_path()+"Al256_reduced/"
param_path = get_data_repo_path()+"example08_data/"
"""
ex08_training_with_postprocessing.py: Uses FESL to first train a network, use this network to predict the LDOS and then
analyze the results of this prediction. This example is structured a little bit different than other examples. 
It consists of two functions, to show that networks can be saved and reaccessed after being trained once.
By default, the training function is commented out. A saved network architecture and parameters is provided in
this repository, so this example focusses more on the analysis part. Nonetheless, the training can also be done 
by simply uncommenting the function call.
Please not that the values calculated at the end will be wrong, because we operate on 0.0025 % of a full DFT simulation
cell for this example. 
"""

# Uses a trained network to make a prediction.
def use_trained_network(network_path, params_path, input_scaler_path, output_scaler_path, doplots, accuracy = 0.05):

    # First we load Parameters and network.
    new_parameters = fesl.Parameters.load_from_file(params_path, no_snapshots=True)

    # Inference should ALWAYS be done with lazy loading activated, even if training was not.
    new_parameters.data.use_lazy_loading = True

    # Now we can build the network.
    new_network = fesl.Network.load_from_file(new_parameters, network_path)

    # We use a data handler object to read the data we want to investigate.
    # We need to make sure that the same scaling is used.
    iscaler = fesl.DataScaler.load_from_file(input_scaler_path)
    oscaler = fesl.DataScaler.load_from_file(output_scaler_path)
    inference_data_handler = fesl.DataHandler(new_parameters, input_data_scaler=iscaler, output_data_scaler=oscaler)

    # Now we can add and load a snapshot to test our new data.
    # Note that we use prepare_data_for_inference instead of the regular prepare_data function.
    new_parameters.data.data_splitting_snapshots = ["te"]
    inference_data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path, "Al_debug_2k_nr2.out.npy", data_path, output_units="1/Ry")
    inference_data_handler.prepare_data()

    # The Tester class is the testing analogon to the training class.
    tester = fesl.Tester(new_parameters)
    tester.set_data(new_network, inference_data_handler)

    # We only have one snapshots, so we are interested in the results of the first snapshot.
    actual_ldos, predicted_ldos = tester.test_snapshot(0)

    # We will use the LDOS calculator to do some preprocessing.
    ldos_calculator = inference_data_handler.target_calculator
    ldos_calculator.read_additional_calculation_data("qe.out", data_path+"QE_Al.scf.pw.out")

    # Get and investigate the DOS.
    actual_dos = ldos_calculator.get_density_of_states(actual_ldos)
    predicted_dos = ldos_calculator.get_density_of_states(predicted_ldos)

    # Plot the DOS.
    if doplots:
        plt.plot(actual_dos, label="actual")
        plt.plot(predicted_dos, label="predicted")
        plt.legend()
        plt.show()

    # Calculate the Band energy.
    # Use a DOS calculator to speed up processing.
    # This is important for bigger (actual) DOS arrays.
    dos_calculator = fesl.DOS.from_ldos(ldos_calculator)
    band_energy_predicted = dos_calculator.get_band_energy(predicted_dos)
    band_energy_actual = dos_calculator.get_band_energy(actual_dos)
    printout("Band energy (actual, predicted, error)[eV]", band_energy_actual, band_energy_predicted, band_energy_predicted-band_energy_actual)
    if np.abs(band_energy_predicted-band_energy_actual) > accuracy:
        return False

    nr_electrons_predicted = dos_calculator.get_number_of_electrons(predicted_dos)
    nr_electrons_actual = dos_calculator.get_number_of_electrons(actual_dos)
    printout("Number of electrons (actual, predicted, error)[eV]", nr_electrons_actual, nr_electrons_predicted, nr_electrons_predicted-nr_electrons_actual)
    if np.abs(band_energy_predicted-band_energy_actual) > accuracy:
        return False
    return True


# Trains a network.
def initial_training(network_path, params_path, input_scaler_path, output_scaler_path, desired_loss_improvement_factor=1):
    ####################
    # PARAMETERS
    # All parameters are handled from a central parameters class that contains subclasses.
    ####################
    test_parameters = fesl.Parameters()
    test_parameters.data.data_splitting_type = "by_snapshot"
    test_parameters.data.data_splitting_snapshots = ["tr", "va", "te"]
    test_parameters.data.input_rescaling_type = "feature-wise-standard"
    test_parameters.data.output_rescaling_type = "normal"
    test_parameters.descriptors.twojmax = 11
    test_parameters.targets.ldos_gridsize = 10
    test_parameters.network.layer_activations = ["ReLU"]
    test_parameters.running.max_number_epochs = 400
    test_parameters.running.mini_batch_size = 40
    test_parameters.running.learning_rate = 0.00001
    test_parameters.running.trainingtype = "Adam"
    test_parameters.targets.ldos_gridsize = 250
    test_parameters.targets.ldos_gridspacing_ev = 0.1
    test_parameters.targets.ldos_gridoffset_ev = -10

    ####################
    # DATA
    # Read data into RAM.
    # We have to specify the directories we want to read the snapshots from.
    # The Handlerinterface will also return input and output scaler objects. These are used internally to scale
    # the data. The objects can be used after successful training for inference or plotting.
    ####################

    data_handler = fesl.DataHandler(test_parameters)

    # Add a snapshot we want to use in to the list.
    data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path, "Al_debug_2k_nr0.out.npy", data_path, output_units="1/Ry")
    data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path, "Al_debug_2k_nr1.out.npy", data_path, output_units="1/Ry")
    data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path, "Al_debug_2k_nr2.out.npy", data_path, output_units="1/Ry")

    data_handler.prepare_data()
    printout("Read data: DONE.")

    ####################
    # NETWORK SETUP
    # Set up the network and trainer we want to use.
    ####################

    test_parameters.network.layer_sizes = [data_handler.get_input_dimension(), 100, data_handler.get_output_dimension()]

    # Setup network and trainer.
    test_network = fesl.Network(test_parameters)
    test_trainer = fesl.Trainer(test_parameters)
    printout("Network setup: DONE.")

    ####################
    # TRAINING
    # Train the network.
    ####################

    printout("Starting training.")
    test_trainer.train_network(test_network, data_handler)
    printout("Training: DONE.")

    ####################
    # SAVING
    # In order to be operational at a later point we need to save 4 objects: Parameters, input/output scaler, network.
    ####################

    test_parameters.save(params_path)
    test_network.save_network(network_path)
    data_handler.input_data_scaler.save(input_scaler_path)
    data_handler.output_data_scaler.save(output_scaler_path)
    if desired_loss_improvement_factor*test_trainer.initial_test_loss < test_trainer.final_test_loss:
        return False
    else:
        return True

def run_example08(dotraining, doinference, doplots=True):
    printout("Welcome to FESL.")
    printout("Running ex08_training_with_postprocessing.py")

    # Choose the paths where the network and the parameters for it should be saved.
    if dotraining is False:
        params_path = param_path+"ex08_params.pkl"
        network_path = param_path+"ex08_network.pth"
        input_scaler_path = param_path+"ex08_iscaler.pkl"
        output_scaler_path = param_path+"ex08_oscaler.pkl"
    else:
        params_path = "./ex08_params.pkl"
        network_path = "./ex08_network.pth"
        input_scaler_path = "./ex08_iscaler.pkl"
        output_scaler_path = "./ex08_oscaler.pkl"


    training_return = True
    inference_return = True
    if dotraining:
        training_return = initial_training(network_path, params_path, input_scaler_path, output_scaler_path)
    if doinference:
        inference_return = use_trained_network(network_path, params_path, input_scaler_path, output_scaler_path, doplots)

    return training_return and inference_return

if __name__ == "__main__":
    if run_example08(False, True):
        printout("Successfully ran ex08_training_with_postprocessing.py.")
    else:
        raise Exception("Ran ex08_training_with_postprocessing but something was off. If you haven't changed any parameters in "
                        "the example, there might be a problem with your installation.")

