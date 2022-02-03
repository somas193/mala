import os
import platform
import time
import statistics
import itertools
import random
import numpy
import scipy
import torch

print(os.cpu_count())
print(f'No. of threads: {torch.get_num_threads()}')
print(f'Processor: {platform.processor()}')
print(f'GPU: {torch.cuda.get_device_name()}')

import pickle
import gc
import mala
from mala import printout
from data_repo_path import data_repo_path
data_path = data_repo_path+"Be2/densities_gp/"

def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_all(3567)

def hypopt_train_test(data_path, use_gpu=False, kernel_choice="rbf", snap_nr=1, ind_pts=500):

    """
    ex16_approx_gaussian_processes_opt.py: Shows how approximate Gaussian processes can be 
    used to learn the electronic density with MALA. Backend is GPytorch.
    Here, an optimization is performed in the sense that the model (hyper-)
    parameters are optimized. It is similar to ex04 in that regard.
    """
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()

    t0_parameters = time.perf_counter()
    params = mala.Parameters()
    params.use_gpu = use_gpu

    # Specify the data scaling.
    params.data.input_rescaling_type = "feature-wise-standard"
    params.data.output_rescaling_type = "normal"

    # Specify the used activation function.
    params.model.variational_dist_type = "cholesky"
    params.model.variational_strategy_type = "variational_strategy"
    params.model.loss_function_type = "gaussian_likelihood"
    params.model.max_log_likelihood = "pll"
    params.model.kernel = kernel_choice

    # Specify the training parameters.
    params.running.max_number_epochs = 20

    # This should be 1, and MALA will set it automatically to, if we don't.
    params.running.mini_batch_size = 1000
    params.running.learning_rate = 0.1
    params.running.trainingtype = "Adam"
    params.targets.target_type = "Density"
    #params.debug.grid_dimensions = [10, 10, 1]
    t1_parameters = time.perf_counter()
    t_parameters = t1_parameters - t0_parameters
    ####################
    # DATA
    # Add and prepare snapshots for training.
    ####################

    t0_datahandler = time.perf_counter()
    data_handler = mala.DataHandler(params)
    inputs_folder = data_path+"inputs_snap/"
    outputs_folder = data_path+"outputs_density/"
    additional_folder = data_path+"additional_info_qeouts/"
    # Add a snapshot we want to use in to the list.
    data_handler.add_snapshot("snapshot0.in.npy", inputs_folder,
                            "snapshot0.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
    if (snap_nr > 1):
        data_handler.add_snapshot("snapshot1.in.npy", inputs_folder,
                                "snapshot1.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
    if (snap_nr > 2):                            
        data_handler.add_snapshot("snapshot2.in.npy", inputs_folder,
                                "snapshot2.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
    data_handler.add_snapshot("snapshot3.in.npy", inputs_folder,
                            "snapshot3.out.npy", outputs_folder, add_snapshot_as="va", output_units="None")
    data_handler.add_snapshot("snapshot4.in.npy", inputs_folder,
                            "snapshot4.out.npy", outputs_folder, add_snapshot_as="te",
                            output_units="None", calculation_output_file=additional_folder+"snapshot4.out")
    data_handler.prepare_data()
    t1_datahandler = time.perf_counter()
    t_datahandler = t1_datahandler - t0_datahandler
    printout("Read data: DONE.")
    inducing_points = data_handler.get_inducing_points(ind_pts)
    

    ####################
    # MODEL SETUP
    # Set up the model and trainer we want to use.
    ####################
    t0_netsetup = time.perf_counter()
    model = mala.ApproxGaussianProcesses(params, inducing_points)
    trainer = mala.Trainer(params, model, data_handler)
    t1_netsetup = time.perf_counter()
    t_netsetup = t1_netsetup - t0_netsetup
    printout("Network setup: DONE.")

    ####################
    # TRAINING
    # Train the network.
    ####################

    printout("Starting training.")
    t0_hypopt = time.perf_counter()
    trainer.train_model()
    t1_hypopt = time.perf_counter()
    t_hypopt = t1_hypopt - t0_hypopt
    printout("Training: DONE.")

    ####################
    # TESTING
    # Pass the first test set snapshot (the test snapshot).
    ####################
    t0_testsetup = time.perf_counter()
    tester = mala.Tester(params, model, data_handler)
    t1_testsetup = time.perf_counter()
    t_testsetup = t1_testsetup - t0_testsetup

    t0_testinf = time.perf_counter()
    actual_density, predicted_density = tester.test_snapshot(0)
    t1_testinf = time.perf_counter()
    t_testinf = t1_testinf - t0_testinf

    density_similarity = scipy.spatial.distance.cosine(actual_density, predicted_density)
    print(f'\nCosine distance between actual and predicted density: {density_similarity}\n')

    # First test snapshot --> 2nd in total
    # data_handler.target_calculator.read_additional_calculation_data("qe.out", data_handler.get_snapshot_calculation_output(4))
    # actual_number_of_electrons = data_handler.target_calculator.get_number_of_electrons(actual_density)
    # predicted_number_of_electrons = data_handler.target_calculator.get_number_of_electrons(predicted_density)
    # printout(f"actual_number_of_electrons: {actual_number_of_electrons}, predicted_number_of_electrons: {predicted_number_of_electrons}")
    del actual_density, predicted_density, tester, trainer, model, data_handler, params
    gc.collect()
    torch.cuda.empty_cache()
    maxmem = torch.cuda.max_memory_reserved()/1073741824

    return maxmem, t_parameters, t_datahandler, t_netsetup, t_hypopt, t_testsetup, t_testinf

dev = ["cpu", "gpu"]
snaps = ["1", "2", "3"]
inducing_pts = ["20", "200", "2000"]
time_types = ["maxmem", "parameters", "datahandler", "netsetup", "hyp_optim", "infsetup", "inference"]
total_types = ['_'.join(f) for f in itertools.product(dev, inducing_pts, snaps, time_types)]
times = {f: [] for f in total_types}

niter = 900

for i in range(niter):
    dev_choice = random.choice(dev)
    if dev_choice == 'cpu':
        use_gpu = False
    if dev_choice == 'gpu':
        use_gpu = True
    snap_nr = random.choice(snaps)
    pts_choice = random.choice(inducing_pts)
    print('\n##########################')
    print('Iteration no.: ', i+1)    
    print(f'Running on: {dev_choice}, inducing points: {pts_choice}, snaps for training: {snap_nr}')
    print('##########################\n')
    
    maxmem, t_parameters, t_datahandler, t_netsetup, t_hypopt, t_testsetup, t_testinf = hypopt_train_test(data_path, use_gpu, snap_nr=int(snap_nr), ind_pts=int(pts_choice))
    times[f'{dev_choice}_{pts_choice}_{snap_nr}_maxmem'].append(maxmem)
    times[f'{dev_choice}_{pts_choice}_{snap_nr}_parameters'].append(t_parameters)
    times[f'{dev_choice}_{pts_choice}_{snap_nr}_datahandler'].append(t_datahandler)
    times[f'{dev_choice}_{pts_choice}_{snap_nr}_netsetup'].append(t_netsetup)
    times[f'{dev_choice}_{pts_choice}_{snap_nr}_hyp_optim'].append(t_hypopt)
    times[f'{dev_choice}_{pts_choice}_{snap_nr}_infsetup'].append(t_testsetup)
    times[f'{dev_choice}_{pts_choice}_{snap_nr}_inference'].append(t_testinf)
    #print(f'{dev_choice}_netsetup : {t_netsetup}')
    #print(f'{dev_choice}_nettrain : {t_nettrain}')

for name, numbers in times.items():
    print('Item:', name, 'Used', len(numbers), 'times')
    print('\tMEDIAN', statistics.median(numbers))
    print('\tMEAN  ', statistics.mean(numbers))
    print('\tSTDEV ', statistics.stdev(numbers))

with open('ApproxGP_hypopt_runtime4.pkl', 'wb') as f:
    pickle.dump(times, f)