from dataset.recsys_dataset_loader import recsys_loader
from dataset.dataset_splitter import datasetsplitter
import os

working_dir = os.path.dirname(os.path.realpath(__file__))
path_to_dataset_dir = working_dir + "/../data/datasets/"
path_to_mappings_dir = working_dir + "/../data/semantic_mappings/"

datasets = {"amazon", "movielens", "movietweetings"}
# datasets = {"movietweetings"}
for dataset in datasets:
    # Load the dataset
    print dataset
    loader = recsys_loader(dataset)
    item_categories = loader.load_item_category_mappings(path_to_mappings_dir)
    print "Categories=" + str(len(item_categories))
    data = loader.load_dataset(path_to_dataset_dir)
    print data

    # Split the dataset
    splitter = datasetsplitter()
    splits = splitter.split_datasets(data, 0.9)
    print splits



