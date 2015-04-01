__author__ = 'rowem'

from dataset.recsys_dataset_loader import recsys_loader
from dataset.dataset_splitter import datasetsplitter
from models.svd import SVD
from learning.sgd_learner import SGD

# Runs a specific parameter configuration over the given model
# set the parameters
params = {"eta": 0.1, "lambda": 0.1, "f": 10, "max_epochs": 100, "epsilon": 0.00001}

path_to_dataset_dir = "/home/rowem/Documents/Git/Data/recsys/datasets/"
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
    print splits

    # train the model using the training split
    svd = SVD(params, splits.train)
    # use SGD to learn for now
    sgd = SGD()
    svd_trained = sgd.train_model(svd, splits.train)
    # apply to the test split
    rmse = sgd.test_model(svd_trained, splits.test)
    print str(rmse)