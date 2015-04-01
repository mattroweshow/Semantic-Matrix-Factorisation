__author__ = 'rowem'

from models.svd_base import baseSVD
from random import shuffle
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

class SGD:

    # trains the model using SGD
    def train_model(self, model, training):
        epoch_count = 1
        while epoch_count <= model.max_epochs:
            reviews = training.reviews
            # shuffle the order of the reviews
            shuffle(reviews)

            # begin stochastic learnin of the model
            for review in reviews:
                rating = model.apply(review)
                error = int(review.rating_score) - rating
                if error is not 0:
                    model.update(review, error)

            print "Learning Epoch done"
            epoch_count += 1

            # break if the model has converged
            if model.convergence_check():
                break

        # write the model's diagnostics
        model.write_diagnostics()

        # return the learnt model
        return model

    # evaluates the model on the held-out test portion
    def test_model(self, model, test):
        actual_ratings = []
        predicted_ratings = []

        # work out the error in prediction for each user and item
        for review in test.reviews:
            predicted_rating = model.apply(review)
            predicted_ratings.append(float(predicted_rating))
            actual_ratings.append(float(review.rating_score))

        rms = sqrt(mean_squared_error(actual_ratings, predicted_ratings))
        return rms