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
    
seed_all(14012022)


def model_train_test(data_path, use_gpu):

    """
    ex01_run_singleshot.py: Shows how a neural models can be trained on material 
    data using this framework. It uses preprocessed data, that is read in 
    from *.npy files.
    """


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

    # Specify the data scaling.
    test_parameters.data.input_rescaling_type = "feature-wise-standard"
    test_parameters.data.output_rescaling_type = "normal"

    # Specify the used activation function.
    test_parameters.model.layer_activations = ["ReLU"]

    # Specify the training parameters.
    test_parameters.running.max_number_epochs = 20
    test_parameters.running.mini_batch_size = 4000
    test_parameters.running.learning_rate = 0.00001
    test_parameters.running.trainingtype = "Adam"
    test_parameters.targets.target_type = "Density"

    test_parameters.use_gpu = use_gpu

    ####################
    # DATA
    # Add and prepare snapshots for training.
    ####################
    t0_datahandler = time.time()
    data_handler = mala.DataHandler(test_parameters)

    # Add a snapshot we want to use in to the list.
    inputs_folder = data_path+"inputs_snap/"
    outputs_folder = data_path+"outputs_density/"
    additional_folder = data_path+"additional_info_qeouts/"
    data_handler.add_snapshot("snapshot0.in.npy", inputs_folder,
                            "snapshot0.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
    data_handler.add_snapshot("snapshot1.in.npy", inputs_folder,
                            "snapshot1.out.npy", outputs_folder, add_snapshot_as="va", output_units="None")
    data_handler.add_snapshot("snapshot2.in.npy", inputs_folder,
                            "snapshot2.out.npy", outputs_folder, add_snapshot_as="te",
                            output_units="None", calculation_output_file=additional_folder+"snapshot2.out")
    data_handler.prepare_data()
    t1_datahandler = time.time()
    t_datahandler = t1_datahandler - t0_datahandler

    ####################
    # NETWORK SETUP
    # Set up the models and trainer we want to use.
    # The layer sizes can be specified before reading data,
    # but it is safer this way.
    ####################
    t0_netsetup = time.time()
    test_parameters.model.layer_sizes = [data_handler.get_input_dimension(),
                                        100,
                                        data_handler.get_output_dimension()]

    # Setup models and trainer.
    test_network = mala.Network(test_parameters)
    test_trainer = mala.Trainer(test_parameters, test_network, data_handler)
    t1_netsetup = time.time()
    t_netsetup = t1_netsetup - t0_netsetup

    ####################
    # TRAINING
    # Train the models.
    ####################
    t0_nettrain = time.time()
    test_trainer.train_model()
    t1_nettrain = time.time()
    t_nettrain = t1_nettrain - t0_nettrain
    
    ####################
    # TESTING
    # Pass the first test set snapshot (the test snapshot).
    ####################
    t0_testsetup = time.time()
    tester = mala.Tester(test_parameters, test_network, data_handler)
    t1_testsetup = time.time()
    t_testsetup = t1_testsetup - t0_testsetup
    
    t0_testinf = time.time()
    actual_density, predicted_density = tester.test_snapshot(0)
    t1_testinf = time.time()
    t_testinf = t1_testinf - t0_testinf
    
    # Do some cleanup
    del actual_density, predicted_density, tester, test_trainer, test_network, data_handler, test_parameters
    gc.collect()
    torch.cuda.empty_cache()
    #print(torch.cuda.memory_summary()) #to make sure that nothing is left on gpu
    
    return t_datahandler, t_netsetup, t_nettrain, t_testsetup, t_testinf


dev = ["cpu", "gpu"]
time_types = ["datahandler", "netsetup", "nettrain", "infsetup", "inference"]
total_types = ['_'.join(f) for f in itertools.product(dev, time_types)]
times = {f: [] for f in total_types}

niter = 200

for i in range(niter):
    dev_choice = random.choice(dev)
    if dev_choice == 'cpu':
        use_gpu = False
    if dev_choice == 'gpu':
        use_gpu = True
    t_datahandler, t_netsetup, t_nettrain, t_testsetup, t_testinf = model_train_test(data_path, use_gpu)
    times[f'{dev_choice}_datahandler'].append(t_datahandler)
    times[f'{dev_choice}_netsetup'].append(t_netsetup)
    times[f'{dev_choice}_nettrain'].append(t_nettrain)
    times[f'{dev_choice}_infsetup'].append(t_testsetup)
    times[f'{dev_choice}_inference'].append(t_testinf)

for name, numbers in times.items():
    print('Item:', name, 'Used', len(numbers), 'times')
    print('\tMEDIAN', statistics.median(numbers))
    print('\tMEAN  ', statistics.mean(numbers))
    print('\tSTDEV ', statistics.stdev(numbers))

with open('runtime.pkl', 'wb') as f:
    pickle.dump(times, f)
