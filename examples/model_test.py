from dataset.recsys_dataset_loader import recsys_loader
from dataset.dataset_splitter import datasetsplitter
from models.svd import SVD
from models.svdplusplus import SVDPlusPlus
from learning.sgd_learner import SGD
import os

# Runs a specific parameter configuration over the given model
# set the parameters
params = {"eta": 0.1, "lambda": 0.1, "f": 10, "max_epochs": 100, "epsilon": 0.00001}

working_dir = os.path.dirname(os.path.realpath(__file__))
path_to_dataset_dir = working_dir + "/../data/datasets/"
datasets = {"amazon", "movielens", "movietweetings"}
# datasets = {"movietweetings"}
for dataset in datasets:
    # Load the dataset
    loader = recsys_loader(dataset)
    data = loader.load_dataset(path_to_dataset_dir)
    print data

    # Split the dataset
    splitter = datasetsplitter()
    splits = splitter.split_datasets(data, 0.9)
    print splits

    # use SGD to learn for now
    sgd = SGD()

    svd = SVD(params, splits.train)
    svd_trained = sgd.train_model(svd, splits.train)
    rmse = sgd.test_model(svd_trained, splits.test)
    print "SVD = " + str(rmse)

    svdplusplus = SVDPlusPlus(params, splits.train)
    svdplusplus_trained = sgd.train_model(svdplusplus, splits.train)
    rmse_svdplusplus = sgd.test_model(svdplusplus_trained, splits.test)
    print "SVD Plus Plus = " + str(rmse_svdplusplus)

