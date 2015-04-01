__author__ = 'rowem'
from operator import itemgetter
from dataset import Dataset, Splits

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










    # have the first split_prop for training and the latter for testing
