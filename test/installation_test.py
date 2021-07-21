import mala
import numpy as np
import os

class TestInstallation:
    """Verifies the installation."""

    def test_installation(self):
        """Test the MALA installation."""
        test_parameters = mala.Parameters()
        test_descriptors = mala.DescriptorInterface(test_parameters)
        test_targets = mala.TargetInterface(test_parameters)
        test_handler = mala.DataHandler(test_parameters,
                                        descriptor_calculator=test_descriptors,
                                        target_calculator=test_targets)
        test_network = mala.Network(test_parameters)
        test_hpoptimizer = mala.HyperOptInterface(test_parameters,
                                                  test_handler)

        # If this test fails, then it will throw an exception way before.
        assert True

    def test_data_repo(self):
        """Test whether the data repo is set up properly"""
        from data_repo_path import get_data_repo_path
        data_path = get_data_repo_path()
        test_array = np.load(os.path.join(data_path, "linking_tester.npy"))
        assert (test_array == [1, 2, 3, 4]).all()