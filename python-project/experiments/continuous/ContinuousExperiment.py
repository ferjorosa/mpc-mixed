from scipy.io import arff
from spn.structure.Base import Context
import pandas as pd
from methods import MSPN, KDE
from abc import ABCMeta, abstractmethod


class ContinuousExperiment(metaclass=ABCMeta):

    def __init__(self, data_name):
        self.data_name = data_name

    @property
    @abstractmethod
    def meta_types(self):
        pass

    @property
    @abstractmethod
    def var_types_string(self):
        pass

    def run(self, run: int, n_folds: int, fold_log: bool, kde_bw: str):
        base_path = "../../../latent_data/continuous/" + self.data_name + "/10_folds/"
        train_datasets = []
        test_datasets = []
        ds_contexts = []

        # Prepare folds' data
        for i in range(1, 11):
            train_data_path = base_path + self.data_name + "_" + str(i) + "_train.arff"
            test_data_path = base_path + self.data_name + "_" + str(i) + "_test.arff"

            # Load data
            train_data = arff.loadarff(train_data_path)
            train_data = pd.DataFrame(train_data[0])
            train_data = train_data.values
            train_datasets.append(train_data)

            test_data = arff.loadarff(test_data_path)
            test_data = pd.DataFrame(test_data[0])
            test_data = test_data.values
            test_datasets.append(test_data)

            # Create context for MSPN algorithm
            ds_context = Context(self.meta_types)
            ds_contexts.append(ds_context)

        # Apply KDE
        results_path = "../../../latent_results/run_" + str(run) + "/continuous/" + self.data_name + "/" + str(n_folds) + "_folds/KDE/"
        KDE.apply(train_datasets, self.var_types_string, test_datasets, n_folds, results_path, self.data_name, fold_log, kde_bw)

        # Apply MSPN
        results_path = "../../../latent_results/run_" + str(run) + "/continuous/" + self.data_name + "/" + str(n_folds) + "_folds/MSPN/"
        MSPN.apply(train_datasets, ds_contexts, test_datasets, n_folds, results_path, self.data_name, fold_log)
