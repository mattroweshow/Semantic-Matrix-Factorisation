__author__ = 'rowem'

class Dataset:
    def __str__(self):
        return "Reviews=" + str(len(self.reviews)) + " | Items=" + str(len(self.items)) + " | Users=" + str(len(self.users))

    def __init__(self, name, reviews, users, items):
        self.name = name
        self.reviews = reviews
        self.users = users
        self.items = items

class Review:
    def __init__(self, userid, itemid, rating_score, time):
        self.userid = userid
        self.itemid = itemid
        self.rating_score = rating_score
        self.time = time

    def __str__(self):
        return str(self.userid) + " | " + str(self.itemid) + " | " + str(self.time) + " | " + str(self.rating_score)

class Item:
    def __init__(self, itemid, title, year):
        self.itemid = itemid
        self.title = title
        self.year = year

class Splits:
    def __init__(self, train, test, split_prop):
        self.train = train
        self.test = test
        self.split_prop = split_prop

    def __str__(self):
        return "Train: " + str(self.train) + " & Test: " + str(self.test)

class Folds:
    def __init__(self, folds):
        self.folds = folds

    def __str__(self):
        output = "["
        for fold in self.folds:
            output += str(len(fold.reviews)) + ", "
        output += "]"
        return output
