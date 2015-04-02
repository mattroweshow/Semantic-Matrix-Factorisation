__author__ = 'mrowe'

from dataset.recsys_dataset_loader import recsys_loader
from dataset.dataset_splitter import datasetsplitter
from tuning.hyperparameter_optimiser import Tuner
from learning.sgd_learner import SGD
from models.svd import SVD
from models.svdplusplus import SVDPlusPlus
import os

# Tell python where to get the data from
working_dir = os.path.dirname(os.path.realpath(__file__))
path_to_dataset_dir = working_dir + "/../data/datasets/"

# Preamble settings
params = {"f": 10, "max_epochs": 100, "epsilon": 0.00001}

# Iterate through each dataset
# datasets = {"amazon", "movielens", "movietweetings"}
datasets = {"movietweetings"}
for dataset in datasets:
    # Load the dataset
    loader = recsys_loader(dataset)
    data = loader.load_dataset(path_to_dataset_dir)
    print data

    # Split the dataset
    splitter = datasetsplitter()
    splits = splitter.split_datasets(data, 0.9)

    f = params['f']

    model_ids = [1, 2]
    for model_id in model_ids:
        # get the tuned hyperparameters for the model and dataset
        tuner = Tuner()
        tuned_hyper_params = tuner.retrieve_tuned_parameters(dataset, model_id, f)
        # Run the model using the training and test folds to compare performance
        params.update(tuned_hyper_params)

        # Set up SGD for learning
        sgd = SGD()

        # SVD
        if model_id is 1:
            svd = SVD(params, splits.train)
            svd_trained = sgd.train_model(svd, splits.train)
            rmse = sgd.test_model(svd_trained, splits.test)
            print "SVD = " + str(rmse)

        # SVD++
        elif model_id is 2:
            svdplusplus = SVDPlusPlus(params, splits.train)
            svdplusplus_trained = sgd.train_model(svdplusplus, splits.train)
            rmse_svdplusplus = sgd.test_model(svdplusplus_trained, splits.test)
            print "SVD Plus Plus = " + str(rmse_svdplusplus)



