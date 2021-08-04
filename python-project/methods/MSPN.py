from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_mspn
import numpy as np
import os
import time
import json
import warnings


def apply(train_datasets, ds_contexts, test_datasets, n_folds, result_path, filename, foldLog):

    # Comment this if you are interested in seen the warnings, we observed that many informative warnings are
    # thrown here, but didn't see nothing suspicious, simly executing Spflow's Mspn method
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("\n========================")
    print("MSPN")
    print("========================")

    results = {}
    folds = {}
    avg_learning_time = 0
    avg_test_ll = 0
    for i in range(1, n_folds + 1):

        index = i-1

        # Only for MSPN:
        ds_contexts[index].add_domains(train_datasets[index])

        init_time = time.time()*1000
        model = learn_mspn(train_datasets[index], ds_contexts[index], min_instances_slice=20)
        end_time = time.time()*1000

        learning_time = end_time - init_time
        test_ll = log_likelihood(model, test_datasets[index])
        test_ll = np.sum(test_ll)

        fold_result = {"test_LL": test_ll, "learning_time": learning_time}

        folds["fold_" + str(i)] = fold_result
        avg_learning_time = avg_learning_time + learning_time
        avg_test_ll = avg_test_ll + test_ll

        if foldLog:
            print("----------------------------------------")
            print("Fold (" + str(i) + "): ")
            print("Test LL: " + str(test_ll))
            print("Learning time: " + str(learning_time))

    # Generate the average results and store them in the dictionary, then store them in a JSON file
    avg_test_ll = avg_test_ll / n_folds
    avg_learning_time = avg_learning_time / n_folds / 1000  # in seconds
    results["average_test_LL"] = avg_test_ll
    results["average_learning_time"] = avg_learning_time
    results["folds"] = folds
    store_json(results, result_path, filename)

    print("----------------------------------------")
    print("----------------------------------------")
    print("Average Test LL: " + str(avg_test_ll))
    print("Average learning time: " + str(avg_learning_time))


def store_json(results, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.isfile(path + filename + "_results_MSPN.json"):
        os.remove(path + filename + "_results_MSPN.json")
        with open(path + filename + "_results_MSPN.json", 'w') as fp:
            json.dump(results, fp, sort_keys=True, indent=4)
    else:
        with open(path + filename + "_results_MSPN.json", 'w') as fp:
            json.dump(results, fp, sort_keys=True, indent=4)
