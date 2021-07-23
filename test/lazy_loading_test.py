from mala.common.parameters import Parameters
from mala.common.printout import printout
from mala.datahandling.data_handler import DataHandler
import torch
import numpy as np
from mala.network.network import Network
from mala.network.trainer import Trainer
from data_repo_path import get_data_repo_path
import time
import pytest
import importlib
import os
# This test compares the data scaling using the regular scaling procedure and
# the lazy-loading one (incremental fitting).

data_path = os.path.join(get_data_repo_path(), "Al36/")
accuracy_strict = 1e-3
accuracy_coarse = 1e-3


class TestLazyLoading:
    """Tests different aspects surrounding lazy loading."""

    def test_scaling(self):
        """
        Test that the scaling works approximately the same for RAM and LL.

        The are some numerical differences that simply occur, but we can still
        check that the scaling results are mostly identical.
        """
        ####################
        # PARAMETERS
        ####################
        test_parameters = Parameters()
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "normal"
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.descriptors.twojmax = 11
        test_parameters.targets.ldos_gridsize = 10
        test_parameters.network.layer_activations = ["LeakyReLU"]
        test_parameters.running.max_number_epochs = 3
        test_parameters.running.mini_batch_size = 512
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.trainingtype = "Adam"
        test_parameters.comment = "Lazy loading test."
        test_parameters.network.nn_type = "feed-forward"
        test_parameters.running.use_gpu = True
        test_parameters.data.use_lazy_loading = False

        ####################
        # DATA
        ####################

        dataset_tester = []
        results = []
        training_tester = []
        for scalingtype in ["standard", "normal", "feature-wise-standard",
                            "feature-wise-normal"]:
            comparison = [scalingtype]
            for ll_type in [True, False]:
                this_result = []
                if ll_type:
                    this_result.append("lazy-loading")
                else:
                    this_result.append("RAM")
                test_parameters.data.use_lazy_loading = ll_type
                test_parameters.data.input_rescaling_type = scalingtype
                test_parameters.data.output_rescaling_type = scalingtype
                data_handler = DataHandler(test_parameters)
                data_handler.clear_data()
                data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path,
                                          "Al_debug_2k_nr0.out.npy", data_path,
                                          output_units="1/Ry",
                                          add_snapshot_as="tr")
                data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path,
                                          "Al_debug_2k_nr1.out.npy", data_path,
                                          output_units="1/Ry",
                                          add_snapshot_as="tr")
                data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                                          "Al_debug_2k_nr2.out.npy", data_path,
                                          output_units="1/Ry",
                                          add_snapshot_as="tr")
                data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path,
                                          "Al_debug_2k_nr1.out.npy", data_path,
                                          output_units="1/Ry",
                                          add_snapshot_as="va")
                data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                                          "Al_debug_2k_nr2.out.npy", data_path,
                                          output_units="1/Ry",
                                          add_snapshot_as="te")
                data_handler.prepare_data()
                if scalingtype == "standard":
                    # The lazy-loading STD equation (and to a smaller amount the
                    # mean equation) is having some small accurcay issue that
                    # I presume to be due to numerical constraints. To make a
                    # meaningful comparison it is wise to scale the value here.
                    this_result.append(data_handler.input_data_scaler.total_mean /
                                       data_handler.nr_training_data)
                    this_result.append(data_handler.input_data_scaler.total_std /
                                       data_handler.nr_training_data)
                    this_result.append(data_handler.output_data_scaler.total_mean /
                                       data_handler.nr_training_data)
                    this_result.append(data_handler.output_data_scaler.total_std /
                                       data_handler.nr_training_data)
                elif scalingtype == "normal":
                    torch.manual_seed(2002)
                    this_result.append(data_handler.input_data_scaler.total_max)
                    this_result.append(data_handler.input_data_scaler.total_min)
                    this_result.append(data_handler.output_data_scaler.total_max)
                    this_result.append(data_handler.output_data_scaler.total_min)
                    dataset_tester.append((data_handler.training_data_set[3998])
                                          [0].sum() +
                                          (data_handler.training_data_set[3999])
                                          [0].sum() +
                                          (data_handler.training_data_set[4000])
                                          [0].sum() +
                                          (data_handler.training_data_set[4001])
                                          [0].sum())
                    test_parameters.network.layer_sizes = \
                        [data_handler.get_input_dimension(), 100,
                         data_handler.get_output_dimension()]

                    # Setup network and trainer.
                    test_network = Network(test_parameters)
                    test_trainer = Trainer(test_parameters, test_network,
                                           data_handler)
                    test_trainer.train_network()
                    training_tester.append(test_trainer.final_test_loss -
                                           test_trainer.initial_test_loss)

                elif scalingtype == "feature-wise-standard":
                    # The lazy-loading STD equation (and to a smaller amount the
                    # mean equation) is having some small accurcay issue that
                    # I presume to be due to numerical constraints. To make a
                    # meaningful comparison it is wise to scale the value here.
                    this_result.append(torch.mean(data_handler.input_data_scaler.
                                                  means)/data_handler.grid_size)
                    this_result.append(torch.mean(data_handler.input_data_scaler.
                                                  stds)/data_handler.grid_size)
                    this_result.append(torch.mean(data_handler.output_data_scaler.
                                                  means)/data_handler.grid_size)
                    this_result.append(torch.mean(data_handler.output_data_scaler.
                                                  stds)/data_handler.grid_size)
                elif scalingtype == "feature-wise-normal":
                    this_result.append(torch.mean(data_handler.input_data_scaler.
                                                  maxs))
                    this_result.append(torch.mean(data_handler.input_data_scaler.
                                                  mins))
                    this_result.append(torch.mean(data_handler.output_data_scaler.
                                                  maxs))
                    this_result.append(torch.mean(data_handler.output_data_scaler.
                                                  mins))

                comparison.append(this_result)
            results.append(comparison)

        for entry in results:
            assert np.isclose(entry[1][1], entry[2][1], atol=accuracy_coarse)
            assert np.isclose(entry[1][2], entry[2][2], atol=accuracy_coarse)
            assert np.isclose(entry[1][3], entry[2][3], atol=accuracy_coarse)
            assert np.isclose(entry[1][4], entry[2][4], atol=accuracy_coarse)
            assert np.isclose(entry[1][1], entry[2][1], atol=accuracy_coarse)
            
        assert np.isclose(dataset_tester[0], dataset_tester[1],
                          atol=accuracy_coarse)
        assert np.isclose(training_tester[0], training_tester[1],
                          atol=accuracy_coarse)

    @pytest.mark.skipif(importlib.util.find_spec("horovod") is None,
                        reason="Horovod is currently not part of the pipeline")
    def test_performance_horovod(self):

        ####################
        # PARAMETERS
        ####################
        test_parameters = Parameters()
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "normal"
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.network.layer_activations = ["LeakyReLU"]
        test_parameters.running.max_number_epochs = 20
        test_parameters.running.mini_batch_size = 500
        test_parameters.running.trainingtype = "Adam"
        test_parameters.comment = "Horovod / lazy loading benchmark."
        test_parameters.network.nn_type = "feed-forward"
        test_parameters.manual_seed = 2021

        ####################
        # DATA
        ####################
        results = []
        for hvduse in [False, True]:
            for ll in [True, False]:
                start_time = time.time()
                test_parameters.running.learning_rate = 0.00001
                test_parameters.data.use_lazy_loading = ll
                test_parameters.use_horovod = hvduse
                data_handler = DataHandler(test_parameters)
                data_handler.clear_data()
                data_handler.add_snapshot("Al_debug_2k_nr0.in.npy",
                                          data_path,
                                          "Al_debug_2k_nr0.out.npy",
                                          data_path,
                                          output_units="1/Ry",
                                          add_snapshot_as="tr")
                data_handler.add_snapshot("Al_debug_2k_nr1.in.npy",
                                          data_path,
                                          "Al_debug_2k_nr1.out.npy",
                                          data_path,
                                          output_units="1/Ry",
                                          add_snapshot_as="tr")
                data_handler.add_snapshot("Al_debug_2k_nr2.in.npy",
                                          data_path,
                                          "Al_debug_2k_nr2.out.npy",
                                          data_path,
                                          output_units="1/Ry",
                                          add_snapshot_as="tr")
                data_handler.add_snapshot("Al_debug_2k_nr1.in.npy",
                                          data_path,
                                          "Al_debug_2k_nr1.out.npy",
                                          data_path,
                                          output_units="1/Ry",
                                          add_snapshot_as="va")
                data_handler.add_snapshot("Al_debug_2k_nr2.in.npy",
                                          data_path,
                                          "Al_debug_2k_nr2.out.npy",
                                          data_path,
                                          output_units="1/Ry",
                                          add_snapshot_as="te")

                data_handler.prepare_data()
                test_parameters.network.layer_sizes = \
                    [data_handler.get_input_dimension(), 100,
                     data_handler.get_output_dimension()]

                # Setup network and trainer.
                test_network = Network(test_parameters)
                test_trainer = Trainer(test_parameters, test_network,
                                       data_handler)
                test_trainer.train_network()

                hvdstring = "no horovod"
                if hvduse:
                    hvdstring = "horovod"

                llstring = "data in RAM"
                if ll:
                    llstring = "using lazy loading"

                results.append([hvdstring, llstring, 
                                test_trainer.initial_test_loss,
                                test_trainer.final_test_loss,
                                time.time() - start_time])

        diff = []
        # For 4 local processes I get:
        # Test:  no horovod ,  using lazy loading
        # Initial loss:  0.1342976689338684
        # Final loss:  0.10587086156010628
        # Time:  3.743736743927002
        # Test:  no horovod ,  data in RAM
        # Initial loss:  0.13430887088179588
        # Final loss:  0.10572846792638302
        # Time:  1.825883388519287
        # Test:  horovod ,  using lazy loading
        # Initial loss:  0.1342976726591587
        # Final loss:  0.10554153844714165
        # Time:  4.513132572174072
        # Test:  horovod ,  data in RAM
        # Initial loss:  0.13430887088179588
        # Final loss:  0.1053303349763155
        # Time:  3.2193074226379395

        for r in results:
            printout("Test: ", r[0], ", ", r[1])
            printout("Initial loss: ", r[2])
            printout("Final loss: ", r[3])
            printout("Time: ", r[4])
            diff.append(r[3] - r[2])
            
        diff = np.array(diff)

        # The loss improvements should be comparable.
        assert np.std(diff) < accuracy_strict
        
