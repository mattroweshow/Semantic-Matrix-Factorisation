__author__ = 'rowem'

from dataset.recsys_dataset_loader import recsys_loader
from dataset.dataset_splitter import datasetsplitter
from tuning.hyperparameter_optimiser import Tuner
import os

# Tell python where to get the data from
working_dir = os.path.dirname(os.path.realpath(__file__))
path_to_dataset_dir = working_dir + "/../data/datasets/"

# Iterate through each dataset
# datasets = {"amazon", "movielens", "movietweetings"}
datasets = {"movietweetings", "movietweetings", "amazon"}
for dataset in datasets:
    # Load the dataset
    loader = recsys_loader(dataset)
    data = loader.load_dataset(path_to_dataset_dir)
    print data

    # Split the dataset: 90:10
    splitter = datasetsplitter()
    splits = splitter.split_datasets(data, 0.9)

    # Split the dataset into 10 for 10-fold cross validation
    folds = splitter.prepare_folds(splits.train, 10)

    # Tune each model
    model_ids = [1, 2]
    for model_id in model_ids:
        # Run the model tuner - best hyperparameters are pushed to the logs directory for the model
        tuner = Tuner()
        params = {"f": 10, "max_epochs": 100, "epsilon": 0.00001}
        hypers = [{"eta": 0.1, "lambda": 0.1}, {"eta": 0.01, "lambda": 0.01}]
        tuner.nfold_cv_hyperparameter_tuner(model_id, params, hypers, folds)
