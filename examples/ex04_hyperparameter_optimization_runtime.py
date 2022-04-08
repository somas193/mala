import os
import platform
import time
import statistics
import itertools
import random
import numpy
import torch

print(os.cpu_count())
print(f'No. of threads: {torch.get_num_threads()}')
print(f'Processor: {platform.processor()}')
print(f'GPU: {torch.cuda.get_device_name()}')

import pickle
import gc
import mala
from mala import printout
from data_repo_path import get_data_repo_path
data_path = get_data_repo_path()+"Be2/densities_gp/"



def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_all(3567)

def hypopt_train_test(data_path, use_gpu=False, snap_nr="1"):

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

    #torch.cuda.synchronize()
    t0_parameters = time.perf_counter()
    test_parameters = mala.Parameters()
    test_parameters.use_gpu = False
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
    test_parameters.running.mini_batch_size = 1000
    test_parameters.running.learning_rate = 0.00001
    test_parameters.running.trainingtype = "Adam"
    test_parameters.targets.target_type = "Density"

    # Specify the number of trials, the hyperparameter optimizer should run
    # and the type of hyperparameter.
    test_parameters.hyperparameters.n_trials = 20
    test_parameters.hyperparameters.hyper_opt_method = "optuna"
    test_parameters.use_gpu = use_gpu
    #torch.cuda.synchronize()
    t1_parameters = time.perf_counter()
    t_parameters = t1_parameters - t0_parameters

    ####################
    # DATA
    # Add and prepare snapshots for training.
    ####################
    #torch.cuda.synchronize()
    t0_datahandler = time.perf_counter()
    data_handler = mala.DataHandler(test_parameters)

    # Add a snapshot we want to use in to the list.
    inputs_folder = data_path+"inputs_snap/"
    outputs_folder = data_path+"outputs_density/"
    additional_folder = data_path+"additional_info_qeouts/"

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
    #torch.cuda.synchronize()
    t1_datahandler = time.perf_counter()
    t_datahandler = t1_datahandler - t0_datahandler
    printout("Read data: DONE.")

    ####################
    # HYPERPARAMETER OPTIMIZATION
    # In order to perform a hyperparameter optimization,
    # one has to simply create a hyperparameter optimizer
    # and let it perform a "study".
    # Before such a study can be done, one has to add all the parameters
    # of interest.
    ####################

    #torch.cuda.synchronize()
    t0_sethypparam = time.perf_counter()
    test_hp_optimizer = mala.HyperOptInterface(test_parameters, data_handler)

    # Learning rate will be optimized.
    test_hp_optimizer.add_hyperparameter("float", "learning_rate",
                                        0.0000001, 0.01)

    # Number of neurons per layer will be optimized.
    test_hp_optimizer.add_hyperparameter("int", "ff_neurons_layer_00", 10, 100)
    test_hp_optimizer.add_hyperparameter("int", "ff_neurons_layer_01", 10, 100)

    # # Batch size will be optimized
    # test_hp_optimizer.add_hyperparameter("int", "mini_batch_size", 40, 4000)

    # Choices for activation function at each layer will be optimized.
    test_hp_optimizer.add_hyperparameter("categorical", "layer_activation_00",
                                        choices=["ReLU", "Sigmoid"])
    test_hp_optimizer.add_hyperparameter("categorical", "layer_activation_01",
                                        choices=["ReLU", "Sigmoid"])
    test_hp_optimizer.add_hyperparameter("categorical", "layer_activation_02",
                                        choices=["ReLU", "Sigmoid"])
    #torch.cuda.synchronize()
    t1_sethypparam = time.perf_counter()
    t_sethypparam = t1_sethypparam - t0_sethypparam

    # Perform hyperparameter optimization.
    printout("Starting Hyperparameter optimization.")
    #torch.cuda.synchronize()
    t0_hypopt = time.perf_counter()
    test_hp_optimizer.perform_study()
    test_hp_optimizer.set_optimal_parameters()
    #torch.cuda.synchronize()
    t1_hypopt = time.perf_counter()
    t_hypopt = t1_hypopt - t0_hypopt
    printout("Hyperparameter optimization: DONE.")
    
    ####################
    # TRAINING
    # Train with these new parameters.
    ####################

    #torch.cuda.synchronize()
    t0_netsetup = time.perf_counter()
    test_network = mala.Network(test_parameters)
    test_trainer = mala.Trainer(test_parameters, test_network, data_handler)
    #torch.cuda.synchronize()
    t1_netsetup = time.perf_counter()
    t_netsetup = t1_netsetup - t0_netsetup 
    printout("Network setup: DONE.")

    #torch.cuda.synchronize()
    t0_nettrain = time.perf_counter()
    test_trainer.train_model()
    #torch.cuda.synchronize()
    t1_nettrain = time.perf_counter()
    t_nettrain = t1_nettrain - t0_nettrain
    printout("Training: DONE.")

    ####################
    # RESULTS.
    # Print the used parameters and check whether the loss decreased enough.
    ####################

    printout("Parameters used for this experiment:")
    test_parameters.show()

    #torch.cuda.synchronize()
    t0_testsetup = time.perf_counter()
    tester = mala.Tester(test_parameters, test_network, data_handler)
    #torch.cuda.synchronize()
    t1_testsetup = time.perf_counter()
    t_testsetup = t1_testsetup - t0_testsetup

    #torch.cuda.synchronize()
    t0_testinf = time.perf_counter()
    actual_density, predicted_density = tester.test_snapshot(0)
    #torch.cuda.synchronize()
    t1_testinf = time.perf_counter()
    t_testinf = t1_testinf - t0_testinf

    # Do some cleanup
    del actual_density, predicted_density, tester, test_trainer, test_network, data_handler, test_parameters, test_hp_optimizer
    gc.collect()
    torch.cuda.empty_cache()
    #print(torch.cuda.memory_summary()) #to make sure that nothing is left on gpu

    return t_parameters, t_datahandler, t_sethypparam, t_hypopt, t_netsetup, t_nettrain, t_testsetup, t_testinf


dev = ["cpu", "gpu"]
snaps = ["1", "2", "3"]
time_types = ["parameters", "datahandler", "sethypparam", "hyp_optim", "netsetup", "nettrain", "infsetup", "inference"]
total_types = ['_'.join(f) for f in itertools.product(dev, snaps, time_types)]
times = {f: [] for f in total_types}

niter = 300

for i in range(niter):
    dev_choice = random.choice(dev)
    if dev_choice == 'cpu':
        use_gpu = False
    if dev_choice == 'gpu':
        use_gpu = True

    snap_nr = random.choice(snaps)
    print('\n##########################')
    print('Iteration no.: ', i+1)    
    print(f'Running on: {dev_choice}, snaps for training: {snap_nr}')
    print('##########################\n')
    
    t_parameters, t_datahandler, t_sethypparam, t_hypopt, t_netsetup, t_nettrain, t_testsetup, t_testinf = hypopt_train_test(data_path, use_gpu, int(snap_nr))
    times[f'{dev_choice}_{snap_nr}_parameters'].append(t_parameters)
    times[f'{dev_choice}_{snap_nr}_datahandler'].append(t_datahandler)
    times[f'{dev_choice}_{snap_nr}_sethypparam'].append(t_sethypparam)
    times[f'{dev_choice}_{snap_nr}_hyp_optim'].append(t_hypopt)
    times[f'{dev_choice}_{snap_nr}_netsetup'].append(t_netsetup)
    times[f'{dev_choice}_{snap_nr}_nettrain'].append(t_nettrain)
    times[f'{dev_choice}_{snap_nr}_infsetup'].append(t_testsetup)
    times[f'{dev_choice}_{snap_nr}_inference'].append(t_testinf)
    print(f'{dev_choice}_{snap_nr}_netsetup : {t_netsetup}')
    print(f'{dev_choice}_{snap_nr}_nettrain : {t_nettrain}')

for name, numbers in times.items():
    print('Item:', name, 'Used', len(numbers), 'times')
    print('\tMEDIAN', statistics.median(numbers))
    print('\tMEAN  ', statistics.mean(numbers))
    print('\tSTDEV ', statistics.stdev(numbers))

with open('NN_hypopt_runtime.pkl', 'wb') as f:
    pickle.dump(times, f)
