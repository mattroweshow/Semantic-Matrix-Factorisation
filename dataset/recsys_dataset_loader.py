__author__ = 'rowem'
import csv
from datetime import datetime
from dataset import Review, Item, Dataset

class recsys_loader:

    def __init__(self, name):
        self.name = name

    # loads the dataset for use
    def load_dataset(self, path_to_dataset_dir):
        # load the review set
        reviews = []
        users = set()
        # read in the file
        afterFirst = False
        with open(path_to_dataset_dir + self.name + '/' + self.name + '_ratings.tsv', 'rb') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            for row in tsvin:
                if len(row) is 4:
                    if afterFirst:
                        userid = row[0]
                        itemid = row[1]
                        rating = row[2]
                        # time = datetime.strptime(row[3], "%Y-%m-%d %H:%M:%S")
                        time = row[3]

                        # compile the review
                        review = Review(userid, itemid, rating, time)
                        reviews.append(review)
                        # add the user
                        users.add(userid)
                    else:
                        afterFirst = True


        # load the item set
        items = []
        # read in the file
        afterFirst = False
        with open(path_to_dataset_dir + self.name + '/' + self.name + '_items.tsv', 'rb') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            for row in tsvin:
                if len(row) is 3:
                    if afterFirst:
                        itemid = row[0]
                        title = row[1]
                        year = int(row[2])

                        # compile the review
                        item = Item(itemid, title, year)
                        items.append(item)
                    else:
                        afterFirst = True

        # Compile the dataset
        dataset = Dataset(self.name, reviews, users, items)
        return dataset

    # Gets the item to category map
    def load_item_category_mappings(self, path_to_mappings_dir):
        item_categories = {}
        # read in the file
        with open(path_to_mappings_dir + self.name + '_cats.tsv', 'rb') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            for row in tsvin:
                if len(row) is 2:
                    # print row
                    itemid = row[0]
                    category = row[1]

                    if itemid in item_categories:
                        categories = item_categories[itemid]
                        categories.append(category)
                        item_categories[itemid] = categories
                    else:
                        item_categories[itemid] = [category]
        return item_categories


