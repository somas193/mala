import mala
from data_repo_path import get_data_repo_path
import numpy as np
import pytest
import importlib
import os

data_path = os.path.join(get_data_repo_path(), "Al36/")
# Control how much the loss should be better after training compared to
# before. This value is fairly high, but we're training on absolutely
# minimal amounts of data.
desired_loss_improvement_factor = 1

# Control the accuracies for the postprocessing routines.
accuracy_electrons = 1e-11
accuracy_total_energy = 1
accuracy_band_energy = 1
accuracy_predictions = 5e-2


class TestFullWorkflow:
    """Tests an entire MALA workflow."""

    def test_network_training(self):
        """Test whether MALA can train a NN."""

        test_trainer = self.__simple_training()
        assert desired_loss_improvement_factor * \
               test_trainer.initial_test_loss > test_trainer.final_test_loss

    @pytest.mark.skipif(importlib.util.find_spec("lammps") is None
                        or os.path.isdir(os.path.join(data_path, "cubes"))
                        is False,
                        reason="LAMMPS is currently not part of the pipeline.")
    def test_preprocessing(self):
        """
        Test whether MALA can preprocess data.

        This means reading the LDOS from cube files and calculating
        SNAP descriptors.
        The data necessary for this is currently not in the data repo!
        """

        # Set up parameters.
        test_parameters = mala.Parameters()
        test_parameters.descriptors.descriptor_type = "SNAP"
        test_parameters.descriptors.twojmax = 10
        test_parameters.descriptors.rcutfac = 4.67637
        test_parameters.data.descriptors_contain_xyz = True
        test_parameters.targets.target_type = "LDOS"
        test_parameters.targets.ldos_gridsize = 10
        test_parameters.targets.ldos_gridspacing_ev = 0.1
        test_parameters.targets.ldos_gridoffset_ev = -10

        # Create a DataConverter, and add snapshots to it.
        data_converter = mala.DataConverter(test_parameters)
        data_converter.add_snapshot_qeout_cube("Al.pw.scf.out", data_path,
                                               "cubes/tmp.pp*Al_ldos.cube",
                                               data_path, output_units="1/Ry")

        data_converter.convert_snapshots("./", naming_scheme="Al_snapshot*")

        # Compare against
        input_data = np.load("Al_snapshot0.in.npy")
        input_data_shape = np.shape(input_data)
        assert input_data_shape[0] == 108 and input_data_shape[1] == 108 and \
               input_data_shape[2] == 100 and input_data_shape[3] == 94

        output_data = np.load("Al_snapshot0.out.npy")
        output_data_shape = np.shape(output_data)
        assert output_data_shape[0] == 108 and output_data_shape[1] == 108 and\
               output_data_shape[2] == 100 and output_data_shape[3] == 10

    def test_postprocessing(self):
        """
        Test whether MALA can postprocess data.

        This means calculating band energy and number of electrons from
        LDOS. Total energy is tested further below.
        """
        # Set up parameters.
        test_parameters = mala.Parameters()
        test_parameters.targets.target_type = "LDOS"
        test_parameters.targets.ldos_gridsize = 250
        test_parameters.targets.ldos_gridspacing_ev = 0.1
        test_parameters.targets.ldos_gridoffset_ev = -10

        # Create a target calculator to perform postprocessing.
        ldos = mala.TargetInterface(test_parameters)
        ldos.read_additional_calculation_data("qe.out",
                                              data_path + "Al.pw.scf.out")
        ldos_data = np.load(data_path + "Al_ldos.npy")

        # Calculate energies
        self_consistent_fermi_energy = ldos. \
            get_self_consistent_fermi_energy_ev(ldos_data)
        number_of_electrons = ldos. \
            get_number_of_electrons(ldos_data, fermi_energy_eV=
                                    self_consistent_fermi_energy)
        band_energy = ldos.get_band_energy(ldos_data,
                                           fermi_energy_eV=
                                           self_consistent_fermi_energy)

        assert np.isclose(number_of_electrons, ldos.number_of_electrons,
                          atol=accuracy_electrons)
        assert np.isclose(band_energy, ldos.band_energy_dft_calculation,
                          atol=accuracy_band_energy)

    @pytest.mark.skipif(importlib.util.find_spec("total_energy") is None,
                        reason="QE is currently not part of the pipeline.")
    def test_total_energy_from_ldos(self):
        """
        Test whether MALA can calculate the total energy using the LDOS.

        This means calculating energies from the LDOS.
        """
        # Set up parameters.
        test_parameters = mala.Parameters()
        test_parameters.targets.target_type = "LDOS"
        test_parameters.targets.ldos_gridsize = 250
        test_parameters.targets.ldos_gridspacing_ev = 0.1
        test_parameters.targets.ldos_gridoffset_ev = -10

        # Create a target calculator to perform postprocessing.
        ldos = mala.TargetInterface(test_parameters)
        ldos.read_additional_calculation_data("qe.out",
                                              data_path + "Al.pw.scf.out")
        ldos_data = np.load(data_path + "Al_ldos.npy")

        # Calculate energies
        self_consistent_fermi_energy = ldos. \
            get_self_consistent_fermi_energy_ev(ldos_data)
        ldos.set_pseudopotential_path(data_path)
        total_energy = ldos.get_total_energy(ldos_data,
                                             fermi_energy_eV=
                                             self_consistent_fermi_energy)
        assert np.isclose(total_energy, ldos.total_energy_dft_calculation,
                          atol=accuracy_total_energy)

    def test_training_with_postprocessing(self):
        """Test if a trained network can be loaded for postprocessing."""
        self.__simple_training(save_network=True)
        self.__use_trained_network()

    def test_training_with_postprocessing_data_repo(self):
        """
        Test if a trained network can be loaded for postprocessing.

        For this test, a network from the data repo will be loaded.
        If this does not work, it's most likely because someting in the MALA
        parameters changed.
        """
        self.__simple_training(save_network=True)
        self.__use_trained_network(get_data_repo_path()+"workflow_test/")

    @staticmethod
    def __simple_training(save_network=False):
        """Perform a simple training and save it, if necessary."""
        # Set up parameters.
        test_parameters = mala.Parameters()
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.data_splitting_snapshots = ["tr", "va", "te"]
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "normal"
        test_parameters.network.layer_activations = ["ReLU"]
        test_parameters.running.max_number_epochs = 400
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.trainingtype = "Adam"

        # Load data.
        data_handler = mala.DataHandler(test_parameters)
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

        # Train a network.
        test_parameters.network.layer_sizes = [
            data_handler.get_input_dimension(),
            100,
            data_handler.get_output_dimension()]

        # Setup network and trainer.
        test_network = mala.Network(test_parameters)
        test_trainer = mala.Trainer(test_parameters, test_network,
                                    data_handler)
        test_trainer.train_network()

        # Save, if necessary.
        if save_network:
            params_path = "workflow_test_params.pkl"
            network_path = "workflow_test_network.pth"
            input_scaler_path = "workflow_test_iscaler.pkl"
            output_scaler_path = "workflow_test_oscaler.pkl"
            test_parameters.save(params_path)
            test_network.save_network(network_path)
            data_handler.input_data_scaler.save(input_scaler_path)
            data_handler.output_data_scaler.save(output_scaler_path)

        return test_trainer

    @staticmethod
    def __use_trained_network(save_path="./"):
        """Use a trained network to make a prediction."""

        params_path = save_path+"workflow_test_params.pkl"
        network_path = save_path+"workflow_test_network.pth"
        input_scaler_path = save_path+"workflow_test_iscaler.pkl"
        output_scaler_path = save_path+"workflow_test_oscaler.pkl"

        # Load parameters, network and data scalers.
        new_parameters = mala.Parameters.load_from_file(params_path,
                                                        no_snapshots=True)
        new_parameters.targets.target_type = "LDOS"
        new_parameters.targets.ldos_gridsize = 250
        new_parameters.targets.ldos_gridspacing_ev = 0.1
        new_parameters.targets.ldos_gridoffset_ev = -10
        new_parameters.data.use_lazy_loading = True
        new_network = mala.Network.load_from_file(new_parameters, network_path)
        iscaler = mala.DataScaler.load_from_file(input_scaler_path)
        oscaler = mala.DataScaler.load_from_file(output_scaler_path)

        # Load data.
        inference_data_handler = mala.DataHandler(new_parameters,
                                                  input_data_scaler=iscaler,
                                                  output_data_scaler=oscaler)
        new_parameters.data.data_splitting_snapshots = ["te"]
        inference_data_handler.add_snapshot("Al_debug_2k_nr2.in.npy",
                                            data_path,
                                            "Al_debug_2k_nr2.out.npy",
                                            data_path,
                                            output_units="1/Ry")
        inference_data_handler.prepare_data(reparametrize_scaler=False)

        # Instantiate and use a Tester object.
        tester = mala.Tester(new_parameters, new_network,
                             inference_data_handler)
        actual_ldos, predicted_ldos = tester.test_snapshot(0)
        ldos_calculator = inference_data_handler.target_calculator
        ldos_calculator.read_additional_calculation_data("qe.out",
                                                         data_path +
                                                         "Al.pw.scf.out")
        band_energy_predicted = ldos_calculator.get_band_energy(predicted_ldos)
        band_energy_actual = ldos_calculator.get_band_energy(actual_ldos)
        nr_electrons_predicted = ldos_calculator.\
            get_number_of_electrons(predicted_ldos)
        nr_electrons_actual = ldos_calculator.\
            get_number_of_electrons(actual_ldos)

        # Check whether the prediction is accurate enough.
        assert np.isclose(band_energy_predicted, band_energy_actual,
                          atol=accuracy_predictions)
        assert np.isclose(nr_electrons_predicted, nr_electrons_actual,
                          atol=accuracy_predictions)