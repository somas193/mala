import torch
import gpytorch
gpytorch.settings.verbose_linalg(True)
gpytorch.settings.cholesky_jitter(float=0, double=0, half=0)
gpytorch.settings.fast_computations(
    covar_root_decomposition=False, log_prob=False, solves=False
)
gpytorch.settings.fast_pred_var(False)
gpytorch.settings.linalg_dtypes(
    default=torch.float64, symeig=torch.float64, cholesky=torch.float64
)
gpytorch.settings.max_cg_iterations(5000)
import mala
import numpy as np
import itertools
import multiprocessing as mp
from sklearn.preprocessing import minmax_scale, scale as standard_scale
from mala import printout
from data_repo_path import data_repo_path
data_path = data_repo_path+"Be2/densities_gp/"

"""
ex12_gassian_processes.py: Shows how Gaussian processes can be used
to learn the electronic density with MALA. Backend is GPytorch.
This is a "Single Shot" Gaussian process, meaning we do not optimize the hyper-
parameters (it is the equivalent to ex01 in that regard.)
"""

params = mala.Parameters()
params.use_gpu = True
#params.manual_seed = 14012022

# Specify the data scaling.
params.data.input_rescaling_type = "feature-wise-standard"
params.data.output_rescaling_type = "normal"

# Specify the used activation function.
params.model.loss_function_type = "gaussian_likelihood"
params.model.kernel = "rbf"
params.model.gp_mean = "constant"

# Specify the training parameters.
params.running.max_number_epochs = 20
params.data.descriptors_contain_xyz = True

# This should be 1, and MALA will set it automatically to, if we don't.
params.running.mini_batch_size = 40
params.running.learning_rate = 0.1
params.running.trainingtype = "Adam"
params.targets.target_type = "Density"
#params.debug.grid_dimensions = [10, 10, 1]
####################
# DATA
# Add and prepare snapshots for training.
####################

data_handler = mala.DataHandler(params)
# Add a snapshot we want to use in to the list.
inputs_folder = "/home/kulkar74/MALA_fork/test-data/Be2/training_data/"
outputs_folder = data_path+"outputs_density/"
additional_folder = data_path+"additional_info_qeouts/"
# Add a snapshot we want to use in to the list.
data_handler.add_snapshot("Be_snapshot1.in.npy", inputs_folder,
                          "snapshot1.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
data_handler.add_snapshot("Be_snapshot2.in.npy", inputs_folder,
                          "snapshot2.out.npy", outputs_folder, add_snapshot_as="va", output_units="None")
data_handler.add_snapshot("Be_snapshot3.in.npy", inputs_folder,
                          "snapshot3.out.npy", outputs_folder, add_snapshot_as="te",
                           output_units="None", calculation_output_file=additional_folder+"snapshot3.out")
data_handler.prepare_data(transpose_data=True)
printout("Read data: DONE.")

####################
# MODEL SETUP
# Set up the model.
# Gaussian Processes do not have to be trained in order
# to captue the trainint data.
####################
model = mala.GaussianProcesses(params, data_handler)
printout("Model Setup: DONE.")
loss = model.compute_loss(*[1, 2e-15])
print(model.covar_module.base_kernel.lengthscale)
print(model.likelihood.noise_covar.noise)
print(loss)
print(*[1, 2e-15])
print([1, 2e-15])

# Hyper param bounds for optimizer
length_scale_bounds = [1e-16, 100]
noise_level_bounds = [1e-16, 100]
# Number of x and y values for plotting. For 50 and better, z(x,y) is
# resolved such that [x_opt_final, y_opt_final] are in the global opt,
# else there might be a little artificial shift.
nsample = 25

# Number of samples for z color scale.
nlevels = 25

# Clip z normalized to [0,1] to that value on order to see differences
# in the z(x,y) landscape.
z_max = 1e-6

x_grid = np.logspace(*np.log10(length_scale_bounds), nsample)
y_grid = np.logspace(*np.log10(noise_level_bounds), nsample)
grid = np.array(list(itertools.product(x_grid, y_grid)))
zz_grid = np.array([model.compute_loss(*i) for i in grid])
print(len(zz_grid))
print("Min loss value: ", np.nanmin(zz_grid))
print("Max loss value: ", np.nanmax(zz_grid))
print(zz_grid)

def scale_and_clip(x, maxval=None):
    """Scale values in x to [0,1], and if maxval is given clip to maxval.

    Modify x in-place.
    """
    msk = np.isnan(x) | np.isinf(x)
    x[msk] = np.nan
    x_cut = x[~msk]
    x_cut = minmax_scale(x_cut)
    if maxval is not None:
        x_cut[x_cut > maxval] = maxval
    x[~msk] = x_cut

scale_and_clip(zz_grid, maxval=z_max)
print("Min loss value: ", np.nanmin(zz_grid))
print("Max loss value: ", np.nanmax(zz_grid))
print(zz_grid)


####################
# TESTING
# Pass the first test set snapshot (the test snapshot).
####################

tester = mala.Tester(params, model, data_handler)
actual_density, predicted_density = tester.test_snapshot(0)

# for dev in range(num_gpus):
# 	print(torch.cuda.memory_summary(f'cuda:{dev}')) # print the cuda memory usage
	
# First test snapshot --> 2nd in total
data_handler.target_calculator.read_additional_calculation_data("qe.out", data_handler.get_snapshot_calculation_output(2))
actual_number_of_electrons = data_handler.target_calculator.get_number_of_electrons(actual_density)
predicted_number_of_electrons = data_handler.target_calculator.get_number_of_electrons(predicted_density)
printout(f"actual_number_of_electrons: {actual_number_of_electrons}, predicted_number_of_electrons: {predicted_number_of_electrons}")

np.save("/home/kulkar74/x_gridpts", x_grid)
np.save("/home/kulkar74/y_gridpts", y_grid)
np.save("/home/kulkar74/z_gridpts", zz_grid)