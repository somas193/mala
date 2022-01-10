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
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_all(5476)

def hypopt_train_test(data_path, use_gpu):

    """
    ex13_gaussian_processes_optimization.py: Shows how Gaussian processes can be 
    used to learn the electronic density with MALA. Backend is GPytorch.
    Here, an optimization is performed in the sense that the model (hyper-)
    parameters are optimized. It is similar to ex04 in that regard.
    """
    torch.cuda.synchronize()
    t0_parameters = time.perf_counter()
    params = mala.Parameters()

    # Specify the data scaling.
    params.data.input_rescaling_type = "feature-wise-standard"
    params.data.output_rescaling_type = "normal"

    # Specify the used activation function.
    params.model.loss_function_type = "gaussian_likelihood"
    params.model.kernel = "rbf"

    # Specify the training parameters.
    params.running.max_number_epochs = 20

    # This should be 1, and MALA will set it automatically to, if we don't.
    params.running.mini_batch_size = 40
    params.running.learning_rate = 0.1
    params.running.trainingtype = "Adam"
    params.targets.target_type = "Density"
    params.use_gpu = use_gpu
    torch.cuda.synchronize()
    t1_parameters = time.perf_counter()
    t_parameters = t1_parameters - t0_parameters
    #params.debug.grid_dimensions = [10, 10, 1]
    ####################
    # DATA
    # Add and prepare snapshots for training.
    ####################

    torch.cuda.synchronize()
    t0_datahandler = time.perf_counter()
    data_handler = mala.DataHandler(params)
    inputs_folder = data_path+"inputs_snap/"
    outputs_folder = data_path+"outputs_density/"
    additional_folder = data_path+"additional_info_qeouts/"
    # Add a snapshot we want to use in to the list.
    data_handler.add_snapshot("snapshot0.in.npy", inputs_folder,
                            "snapshot0.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
    # data_handler.add_snapshot("snapshot1.in.npy", inputs_folder,
    #                         "snapshot1.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
    # data_handler.add_snapshot("snapshot2.in.npy", inputs_folder,
    #                         "snapshot2.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
    data_handler.add_snapshot("snapshot1.in.npy", inputs_folder,
                            "snapshot1.out.npy", outputs_folder, add_snapshot_as="va", output_units="None")
    data_handler.add_snapshot("snapshot2.in.npy", inputs_folder,
                            "snapshot2.out.npy", outputs_folder, add_snapshot_as="te",
                            output_units="None", calculation_output_file=additional_folder+"snapshot2.out")
    data_handler.prepare_data(transpose_data=True)
    torch.cuda.synchronize()
    t1_datahandler = time.perf_counter()
    t_datahandler = t1_datahandler - t0_datahandler
    printout("Read data: DONE.")

    ####################
    # MODEL SETUP
    # Set up the model and trainer we want to use.
    ####################

    torch.cuda.synchronize()
    t0_netsetup = time.perf_counter()
    model = mala.GaussianProcesses(params, data_handler)
    trainer = mala.Trainer(params, model, data_handler)
    torch.cuda.synchronize()
    t1_netsetup = time.perf_counter()
    t_netsetup = t1_netsetup - t0_netsetup
    printout("Network setup: DONE.")

    ####################
    # TRAINING
    # Train the network.
    ####################

    printout("Starting training.")
    torch.cuda.synchronize()
    t0_hypopt = time.perf_counter()
    trainer.train_model()
    torch.cuda.synchronize()
    t1_hypopt = time.perf_counter()
    t_hypopt = t1_hypopt - t0_hypopt
    printout("Training: DONE.")

    ####################
    # TESTING
    # Pass the first test set snapshot (the test snapshot).
    ####################
    torch.cuda.synchronize()
    t0_testsetup = time.perf_counter()
    tester = mala.Tester(params, model, data_handler)
    torch.cuda.synchronize()
    t1_testsetup = time.perf_counter()
    t_testsetup = t1_testsetup - t0_testsetup

    torch.cuda.synchronize()
    t0_testinf = time.perf_counter()
    actual_density, predicted_density = tester.test_snapshot(0)
    torch.cuda.synchronize()
    t1_testinf = time.perf_counter()
    t_testinf = t1_testinf - t0_testinf

    # Do some cleanup
    del actual_density, predicted_density, tester, trainer, model, data_handler, params
    gc.collect()
    torch.cuda.empty_cache()
    #print(torch.cuda.memory_summary()) #to make sure that nothing is left on gpu

    return t_parameters, t_datahandler, t_netsetup, t_hypopt, t_testsetup, t_testinf

dev = ["cpu", "gpu"]
time_types = ["parameters", "datahandler", "netsetup", "hyp_optim", "infsetup", "inference"]
total_types = ['_'.join(f) for f in itertools.product(dev, time_types)]
times = {f: [] for f in total_types}

niter = 80

for i in range(niter):
    dev_choice = random.choice(dev)
    if dev_choice == 'cpu':
        use_gpu = False
    if dev_choice == 'gpu':
        use_gpu = True
    print('\n##########################')
    print('Iteration no.: ', i+1)    
    print('Running on: ', dev_choice)
    print('##########################\n')
    
    t_parameters, t_datahandler, t_netsetup, t_hypopt, t_testsetup, t_testinf = hypopt_train_test(data_path, use_gpu)
    times[f'{dev_choice}_parameters'].append(t_parameters)
    times[f'{dev_choice}_datahandler'].append(t_datahandler)
    times[f'{dev_choice}_netsetup'].append(t_netsetup)
    times[f'{dev_choice}_hyp_optim'].append(t_hypopt)
    times[f'{dev_choice}_infsetup'].append(t_testsetup)
    times[f'{dev_choice}_inference'].append(t_testinf)
    #print(f'{dev_choice}_netsetup : {t_netsetup}')
    #print(f'{dev_choice}_nettrain : {t_nettrain}')

for name, numbers in times.items():
    print('Item:', name, 'Used', len(numbers), 'times')
    print('\tMEDIAN', statistics.median(numbers))
    print('\tMEAN  ', statistics.mean(numbers))
    print('\tSTDEV ', statistics.stdev(numbers))

with open('GP_hypopt_runtime.pkl', 'wb') as f:
    pickle.dump(times, f)