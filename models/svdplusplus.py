__author__ = 'rowem'
from svd_base import baseSVD
import numpy as np
import math

class SVDPlusPlus(baseSVD):
    def __init__(self, parameters, training):
        # print "priming the model"
        # set the learning params
        self.max_epochs = parameters['max_epochs']
        self.epsilon = parameters['epsilon']

        # set the hyperparams
        self.eta = parameters['eta']
        self.lambd = parameters['lambda']
        self.f = parameters['f']

        # set the training data
        self.training = training

        # prime the model logging components
        self.errors = []

        # initialise the biases
        self.item_biases = {}
        self.prior_item_biases = {}
        self.user_biases = {}
        self.prior_user_biases = {}

        # initialise the latent factor vectors
        self.item_latent_factors = {}
        self.prior_item_latent_factors = {}
        self.user_latent_factors = {}
        self.prior_user_latent_factors = {}

        # Prime the user specific properties
        self.user_rated_items = {}
        self.rated_item_latent_factors = {}
        self.prior_rated_item_latent_factors = {}

        # prime the average rating
        self.mu = self.derive_average_rating()

        # prime the biases and latent factor vectors
        for itemid in training.items:
            self.item_biases[itemid] = 0
            self.item_latent_factors[itemid] = np.zeros(self.f)
            self.rated_item_latent_factors[itemid] = np.zeros(self.f)
        for userid in training.users:
            self.user_biases[userid] = 0
            self.user_latent_factors[userid] = np.zeros(self.f)
        for review in training.reviews:
            if review.userid in self.user_rated_items:
                rated_items = self.user_rated_items[review.userid]
                rated_items.add(review.itemid)
                self.user_rated_items[review.userid] = rated_items
            else:
                rated_items = set()
                rated_items.add(review.itemid)
                self.user_rated_items[review.userid] = rated_items

        self.epochs = 0

    # returns the predicted rating of the review
    def apply(self, review):
        # print "applying the model"
        predicted_rating = self.mu

        # if we have the user and item bias then use them
        if review.itemid in self.item_biases and review.userid in self.user_biases:
            item_bias = self.item_biases[review.itemid]
            user_bias = self.user_biases[review.userid]

            item_latent_factors = self.item_latent_factors[review.itemid]
            user_latent_factors = self.user_latent_factors[review.userid]

            rated_items = self.user_rated_items[review.userid]
            # prime the rated latent factor vectors
            scalar = 1 / math.sqrt(len(rated_items))
            sum_rated_latent_factors = np.zeros(self.f)
            for rated_item in rated_items:
                sum_rated_latent_factors += self.rated_item_latent_factors[rated_item]
            sum_rated_latent_factors *= scalar

            # compute the dot product between the item latent factor vector and the user latent factor vector
            personal_comp = np.dot(item_latent_factors, (user_latent_factors + sum_rated_latent_factors))

            # determine the predicted rating
            predicted_rating += item_bias + user_bias + personal_comp

          # if we only have item bias then use that
        elif review.itemid in self.item_biases and review.userid not in self.user_biases:
            item_bias = self.item_biases[review.itemid]

            # determine the predicted rating
            predicted_rating += item_bias

            # if we only have the user bias then use that
        elif review.itemid not in self.item_biases and review.userid in self.user_biases:
            user_bias = self.user_biases[review.userid]

            # determine the predicted rating
            predicted_rating += user_bias

        # print predicted_rating

        return predicted_rating

    def update(self, review, error):

        if review.itemid in self.item_biases and review.userid in self.user_biases:
            # update the biases
            item_bias = self.item_biases[review.itemid]
            self.item_biases[review.itemid] = item_bias + self.eta * (error - self.lambd * item_bias)

            user_bias = self.user_biases[review.userid]
            self.user_biases[review.userid] = user_bias + self.eta * (error - self.lambd * user_bias)

            # update the latent factor vectors
            item_latent_factors = self.item_latent_factors[review.itemid]
            old_item_latent_factors = item_latent_factors
            user_latent_factors = self.user_latent_factors[review.userid]
            old_user_latent_factors = user_latent_factors
            rated_items = self.user_rated_items[review.userid]
            old_rated_item_latent_factors = self.rated_item_latent_factors

            # we now incorporate the rated latent factors within the model
            scalar = 1 / math.sqrt(len(rated_items))
            sum_rated_latent_factors = np.zeros(self.f)
            for rated_item in rated_items:
                sum_rated_latent_factors += self.rated_item_latent_factors[rated_item]
            sum_rated_latent_factors *= scalar
            item_latent_factors += self.eta * (error * (old_user_latent_factors + sum_rated_latent_factors)
                                               - self.lambd * old_item_latent_factors)
            user_latent_factors += self.eta * (error * old_item_latent_factors
                                               - self.lambd * old_user_latent_factors)

            for rated_item in rated_items:
                self.rated_item_latent_factors[rated_item] += self.eta * ((error * scalar * old_item_latent_factors)
                                                                          - self.lambd * self.rated_item_latent_factors[rated_item])

            self.item_latent_factors[review.itemid] = item_latent_factors
            self.user_latent_factors[review.userid] = user_latent_factors

        # log the error
        self.errors.append(error)


    def write_diagnostics(self):
        # f = open('myfile', 'w')
        # for error in self.errors:
        #     f.write(str(error) + '\n')
        # f.close()
        pass


    def derive_average_rating(self):
        average_rating = sum([int(review.rating_score) for review in self.training.reviews]) / len(self.training.reviews)
        return average_rating

    def convergence_check(self):

        converged = True

        # first epoch just set the priors to be the current model params
        if len(self.prior_item_biases) is 0:
            self.prior_item_biases = self.item_biases
            self.prior_user_biases = self.user_biases
            self.prior_item_latent_factors = self.item_latent_factors
            self.prior_user_latent_factors = self.user_latent_factors
            self.prior_rated_item_latent_factors = self.rated_item_latent_factors
            converged = False

        else:
            # check item biases and latent factors for convergence
            for itemid in self.item_biases:
                if abs(self.item_biases[itemid] - self.prior_item_biases[itemid]) > self.epsilon:
                    converged = False
                    break

                latent_factor_vector = self.item_latent_factors[itemid]
                prior_factor_vector = self.prior_item_latent_factors[itemid]
                for i in range(0, self.f):
                    if abs(latent_factor_vector[i] - prior_factor_vector[i]) > self.epsilon:
                        converged = False
                        break

                rated_latent_factor_vector = self.rated_item_latent_factors[itemid]
                prior_rated_latent_factor_vector = self.prior_rated_item_latent_factors[itemid]
                for i in range(0, self.f):
                    if abs(rated_latent_factor_vector[i] - prior_rated_latent_factor_vector[i]) > self.epsilon:
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

        return converged



