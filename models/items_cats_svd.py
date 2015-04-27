from svd_base import baseSVD
import numpy as np
import os
import math
import sys

class ItemsCatsSVD(baseSVD):
    def __init__(self, parameters, training, item_cats, verbose):
        print "Initialising the model"

        # set the learning params
        self.max_epochs = parameters['max_epochs']
        self.epsilon = parameters['epsilon']

        # set the hyperparams
        self.eta = parameters['eta']
        self.lambd = parameters['lambda']
        # Leave this fixed for now
        self.f = parameters['f']

        # set the training data
        self.training = training

        # set the item categories
        self.item_cats = item_cats

        self.verbose = verbose

        # prime the model logging components
        self.errors = []

        # initialise the biases
        self.item_biases = {}
        self.prior_item_biases = {}
        self.user_biases = {}
        self.prior_user_biases = {}

        # initialise the latent factor vectors for the users
        self.user_latent_factors = {}
        self.prior_user_latent_factors = {}

        # initialise the observed category items ratings
        self.item_cats_ratings = {}

        # prime the average rating
        self.mu = self.derive_average_rating()

        # prime the users' latent factors
        for userid in training.users:
            self.user_biases[userid] = 0
            self.user_latent_factors[userid] = np.zeros(self.f)

        # prime the item biases
        for itemid in training.items:
            self.item_biases[itemid] = 0


        # prime the categories' latent factors
        self.unique_cats = []
        for item in item_cats:
            cats = item_cats[item]
            for cat in cats:
                if cat not in self.unique_cats:
                    self.unique_cats.append(cat)
        if self.verbose:
            print "Unique Cats = " + str(len(self.unique_cats))
#        for cat in unique_cats:
#            self.cats_latent_factors[cat] = np.zeros(self.f)

        # create a matrix version of this
        self.cats_latent_factors_matrix = np.ndarray(shape=(len(self.unique_cats), self.f), dtype = float)
        self.prior_cats_latent_factors_matrix = np.ndarray(shape=(len(self.unique_cats), self.f), dtype = float)

        # prime the average category rating
        for item in item_cats:
            item_cat_avg_rating_vector = np.zeros(len(self.unique_cats))
            cats = item_cats[item]

            # get the ratings of this item
            item_reviews = [review for review in training.reviews if review.itemid is item]

            # set the average category rating for each vector element
            avg_rating = 0
            if len(item_reviews) is not 0:
                avg_rating = sum([int(review.rating_score) for review in item_reviews]) / len(item_reviews)
            for cat in cats:
                item_cat_avg_rating_vector[self.unique_cats.index(cat)] = avg_rating

            # update the item to cats ratings
            self.item_cats_ratings[item] = item_cat_avg_rating_vector

        self.epochs = 0

    def reset_hyperparameters(self, hypers):
        self.eta = hypers['eta']
        self.lambd = hypers['lambda']

        # returns the predicted rating of the review
    def apply(self, review):
#        print "deriving rating"
        predicted_rating = self.mu

        # if we have the user and item bias and the item has been mapped to categories then use them
        if review.itemid in self.item_biases and review.userid in self.user_biases and review.itemid in self.item_cats_ratings:
            item_bias = self.item_biases[review.itemid]
            user_bias = self.user_biases[review.userid]

            user_latent_factors = self.user_latent_factors[review.userid]
            item_categories_ratings = self.item_cats_ratings[review.itemid]

            # compute the dot product between the user latent factor vector and the categories by latent factors matrix
            pc_1 = np.dot(user_latent_factors, np.transpose(self.cats_latent_factors_matrix))
            pc_2 = np.dot(pc_1, np.transpose(item_categories_ratings))

            # Print everything if verbose is called
            if self.verbose:
                print "User Latent Factors = " + str(user_latent_factors)
                print str(user_latent_factors.shape)
                print "Item Category Ratings = " + str(item_categories_ratings)
                print str(item_categories_ratings.shape)
                print "Cats latent factors matrix = " + str(self.cats_latent_factors_matrix)
                print str(self.cats_latent_factors_matrix.shape)
                print "PC1 = " + str(pc_1)
                print str(pc_1.shape)
                print "PC2 = " + str(pc_2)
                print str(pc_2.shape)


            # determine the predicted rating
            predicted_rating += item_bias + user_bias + pc_2
            output = "\nPC_1 = " + str(pc_1)
            output += "\nPC_2=" + str(pc_2)
            output += "\nPredicted rating = " + str(predicted_rating)
            self.log_output(output)

        elif review.itemid in self.item_biases and review.userid in self.user_biases and review.itemid not in self.item_cats_ratings:
            # if we only have item bias then use that
            item_bias = self.item_biases[review.itemid]
            user_bias = self.user_biases[review.userid]
            predicted_rating += item_bias + user_bias

        elif review.itemid in self.item_biases and review.userid not in self.user_biases:
            item_bias = self.item_biases[review.itemid]

            # determine the predicted rating
            predicted_rating += item_bias

            # if we only have the user bias then use that
        elif review.itemid not in self.item_biases and review.userid in self.user_biases:
            user_bias = self.user_biases[review.userid]

            # determine the predicted rating
            predicted_rating += user_bias

        if math.isnan(predicted_rating):
            # print self.item_biases
            # print self.user_latent_factors[review.userid]
            # print self.cats_latent_factors_matrix
            sys.exit()

        return predicted_rating

    def update(self, review, error):
        # print str(error)

        if review.itemid in self.item_biases and review.userid in self.user_biases and review.itemid in self.item_cats_ratings:
            # update the biases
            item_bias = self.item_biases[review.itemid]
            self.item_biases[review.itemid] = item_bias + self.eta * (error - self.lambd * item_bias)

            user_bias = self.user_biases[review.userid]
            self.user_biases[review.userid] = user_bias + self.eta * (error - self.lambd * user_bias)

            # update the latent factor vectors
            user_latent_factors = self.user_latent_factors[review.userid]
            old_user_latent_factors = user_latent_factors
            cats_m = self.cats_latent_factors_matrix
            old_cats_m = cats_m

            # update the user latent factors
            item_categories_ratings = self.item_cats_ratings[review.itemid]
            user_latent_factors += self.eta * (error * np.dot(item_categories_ratings, cats_m)
                                               - self.lambd * old_user_latent_factors)

            # update the categories' latent factors
            for cat in self.item_cats[review.itemid]:
                cats_m[self.unique_cats.index(cat)] += self.eta * \
                                                       (error * self.user_latent_factors[review.userid]
                                                        * self.item_cats_ratings[review.itemid][self.unique_cats.index(cat)]
                                                        - self.lambd * old_cats_m[self.unique_cats.index(cat)])

            # reset the latent factors
            self.user_latent_factors[review.userid] = user_latent_factors
            self.cats_latent_factors_matrix = cats_m

        elif review.itemid in self.item_biases and review.userid in self.user_biases and review.itemid in self.item_cats_ratings:
            # update the biases
            item_bias = self.item_biases[review.itemid]
            self.item_biases[review.itemid] = item_bias + self.eta * (error - self.lambd * item_bias)

            user_bias = self.user_biases[review.userid]
            self.user_biases[review.userid] = user_bias + self.eta * (error - self.lambd * user_bias)

        # log the error
        self.errors.append(error)


    def write_diagnostics(self):
        print "Item Biases = " + str(self.item_biases)
        print "User Biases = " + str(self.user_biases)
        print "User LF Matrix = " + str(self.user_latent_factors)
        print "Cats LF Matrix = " + str(self.cats_latent_factors_matrix)


    def derive_average_rating(self):
        average_rating = sum([int(review.rating_score) for review in self.training.reviews]) / len(self.training.reviews)
        return average_rating

    def convergence_check(self):

        converged = True

        # first epoch just set the priors to be the current model params
        if len(self.prior_item_biases) is 0:
            self.prior_item_biases = self.item_biases
            self.prior_user_biases = self.user_biases
            self.prior_user_latent_factors = self.user_latent_factors
            self.prior_cats_latent_factors_matrix = self.cats_latent_factors_matrix
            converged = False

        else:
            # check item biases and latent factors for convergence
            for itemid in self.item_biases:
                if abs(self.item_biases[itemid] - self.prior_item_biases[itemid]) > self.epsilon:
                    converged = False
                    break

            # check the user biases and latent factors for convergence
            for userid in self.user_biases:
                if abs(self.user_biases[userid] - self.prior_user_biases[userid]) > self.epsilon:
                    converged = False
                    break

                latent_factor_vector = self.user_latent_factors[userid]
                prior_factor_vector = self.prior_user_latent_factors[userid]
                for i in range(0, self.f):
                    if abs(latent_factor_vector[i] - prior_factor_vector[i]) > self.epsilon:
                        converged = False
                        break

            # check that the category factors have converged: compare category wise vectors
            for row_num in range(0, self.cats_latent_factors_matrix.shape[0], 1):
                row = self.cats_latent_factors_matrix[row_num, :]
                old_row = self.prior_cats_latent_factors_matrix[row_num, :]
                if abs(sum(row + old_row)) > self.epsilon:
                    converged = False
                    break

        return converged

    def log_output(self, output):
        working_dir = os.path.dirname(os.path.realpath(__file__))
        log_file_path = working_dir + "/../data/log/log_file_item_cats_svd.log"
        with open(log_file_path, "a") as myfile:
            myfile.write(output)