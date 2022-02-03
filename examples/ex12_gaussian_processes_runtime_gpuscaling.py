import time
import statistics
import itertools
import random
import numpy
import torch
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 14012022
random.seed(seed)


def model_train_test(data_path, use_gpu, gpu_nr):
    
    seed_all(seed)
    torch.cuda.reset_peak_memory_stats()

    """
    ex12_gassian_processes.py: Shows how Gaussian processes can be used
    to learn the electronic density with MALA. Backend is GPytorch.
    This is a "Single Shot" Gaussian process, meaning we do not optimize the hyper-
    parameters (it is the equivalent to ex01 in that regard.)
    """

    params = mala.Parameters()
    params.manual_seed = seed
    # Specify the data scaling.
    params.data.input_rescaling_type = "feature-wise-standard"
    params.data.output_rescaling_type = "normal"

    # Specify the used activation function.
    params.model.loss_function_type = "gaussian_likelihood"
    params.model.kernel = "rbf"

    # Specify the training parameters.
    params.running.max_number_epochs = 20

    # This should be 1, and MALA will set it automatically to, if we don't.
    params.running.mini_batch_size = 1
    params.running.learning_rate = 0.1
    params.running.trainingtype = "Adam"
    params.targets.target_type = "Density"
    #params.debug.grid_dimensions = [10, 10, 1]

    params.use_gpu = use_gpu

    ####################
    # DATA
    # Add and prepare snapshots for training.
    ####################
    data_handler = mala.DataHandler(params)
    inputs_folder = data_path+"inputs_snap/"
    outputs_folder = data_path+"outputs_density/"
    additional_folder = data_path+"additional_info_qeouts/"
    # Add a snapshot we want to use in to the list.
    data_handler.add_snapshot("snapshot0.in.npy", inputs_folder,
                            "snapshot0.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
    data_handler.add_snapshot("snapshot1.in.npy", inputs_folder,
                            "snapshot1.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
    data_handler.add_snapshot("snapshot2.in.npy", inputs_folder,
                            "snapshot2.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
                            
    data_handler.add_snapshot("snapshot3.in.npy", inputs_folder,
                            "snapshot3.out.npy", outputs_folder, add_snapshot_as="va", output_units="None")
    data_handler.add_snapshot("snapshot4.in.npy", inputs_folder,
                            "snapshot4.out.npy", outputs_folder, add_snapshot_as="te",
                            output_units="None", calculation_output_file=additional_folder+"snapshot4.out")
    data_handler.prepare_data(transpose_data=True)

    ####################
    # MODEL SETUP
    # Set up the model.
    # Gaussian Processes do not have to be trained in order
    # to captue the trainint data.
    ####################
    t0_netsetup = time.time()
    model = mala.GaussianProcesses(params, data_handler, num_gpus=gpu_nr)
    t1_netsetup = time.time()
    t_netsetup = t1_netsetup - t0_netsetup

    ####################
    # TESTING
    # Pass the first test set snapshot (the test snapshot).
    ####################
    t0_testsetup = time.time()
    tester = mala.Tester(params, model, data_handler)
    t1_testsetup = time.time()
    t_testsetup = t1_testsetup - t0_testsetup

    t0_testinf = time.time()
    actual_density, predicted_density = tester.test_snapshot(0)
    t1_testinf = time.time()
    t_testinf = t1_testinf - t0_testinf

    # Do some cleanup
    del actual_density, predicted_density, tester, model, data_handler, params
    gc.collect()
    torch.cuda.empty_cache()
    #print(torch.cuda.memory_summary()) #to make sure that nothing is left on gpu
    
    dev_mem_usage = []
    for dev_nr in range(gpu_nr):
        dev_mem_usage.append(torch.cuda.max_memory_allocated(f'cuda:{dev_nr}'))
    
    return torch.cuda.max_memory_allocated('cuda:0'), t_netsetup, t_testsetup, t_testinf, dev_mem_usage

dev = ["gpu"]
gpus = ["1", "2", "3"]
time_types = ["maxmem", "netsetup", "infsetup", "inference"]
total_types = ['_'.join(f) for f in itertools.product(dev, gpus, time_types)]
times = {f: [] for f in total_types}
mem_usage_array = numpy.zeros((3, 3))

niter = 100

for i in range(niter):
    dev_choice = 'gpu'
    use_gpu = True

    gpu_nr = random.choice(gpus)
    print(f'\tRunning on: {dev_choice}, gpus for training: {gpu_nr}')
    
    maxmem, t_netsetup, t_testsetup, t_testinf, dev_mem_usage = model_train_test(data_path, use_gpu, int(gpu_nr))
    times[f'{dev_choice}_{gpu_nr}_maxmem'].append(maxmem)
    times[f'{dev_choice}_{gpu_nr}_netsetup'].append(t_netsetup)
    times[f'{dev_choice}_{gpu_nr}_infsetup'].append(t_testsetup)
    times[f'{dev_choice}_{gpu_nr}_inference'].append(t_testinf)
    #print(f'{dev_choice}_{gpu_nr}_netsetup : {t_netsetup}')
    #print(f'{dev_choice}_{gpu_nr}_inference : {t_testinf}')
    
    for j in range(int(gpu_nr)):
        mem_usage_array[int(gpu_nr) - 1][j] = dev_mem_usage[j]

for name, numbers in times.items():
    print('Item:', name, 'Used', len(numbers), 'times')
    print('\tMEDIAN', statistics.median(numbers))
    print('\tMEAN  ', statistics.mean(numbers))
    print('\tSTDEV ', statistics.stdev(numbers))

with open('GP_runtime_gpuscaling.pkl', 'wb') as f:
    pickle.dump(times, f)

numpy.savetxt('GP_runtime_gpuscaling_mem.txt', mem_usage_array)
