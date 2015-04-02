__author__ = 'rowem'
from operator import itemgetter
from dataset import Dataset, Splits, Folds
from random import shuffle
import numpy as np

class datasetsplitter:
    def __init__(self):
        pass

    def split_datasets(self, dataset, split_prop):

        # get the reviews
        reviews = dataset.reviews

        # order the reviews by their date
        ordered_reviews = sorted(reviews, key=lambda x: x.time)

        # split by the first split_prop of the dataset
        split_point = int(split_prop * len(ordered_reviews))
        training_reviews = ordered_reviews[:split_point]
        training_users = set([review.userid for review in training_reviews])
        training_items = set([review.itemid for review in training_reviews])
        test_reviews = ordered_reviews[split_point:]
        test_users = set([review.userid for review in test_reviews])
        test_items = set([review.itemid for review in test_reviews])

        # prep the datasets
        training = Dataset(dataset.name, training_reviews, training_users, training_items)
        test = Dataset(dataset.name, test_reviews, test_users, test_items)

        # prep the splits
        splits = Splits(training, test, split_prop)
        return splits

    # Splits the dataset's reviews into #fold_count folds for cross-validation
    def prepare_folds(self, dataset, fold_count):
        # get the reviews and shuffle their order
        reviews = dataset.reviews
        shuffle(reviews)

        # get the indices of the folds
        review_indices = range(0, len(reviews))
        review_ndarray = np.asarray(review_indices)
        n_lists = np.array_split(review_ndarray, fold_count)

        # Create the folds as a list of dataset objects
        folds_list = []
        for i in range(0, fold_count):
            fold_reviews = reviews[n_lists[i][0]:n_lists[i][len(n_lists[i])-1]]
            fold_users = set([review.userid for review in fold_reviews])
            fold_items = set([review.itemid for review in fold_reviews])
            fold_dataset = Dataset(dataset.name, fold_reviews, fold_users, fold_items)
            folds_list.append(fold_dataset)

        # prepare the folds to be written back
        folds = Folds(folds_list)
        return folds