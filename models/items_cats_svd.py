from svd_base import baseSVD
import numpy as np

class ItemsCatsSVD(baseSVD):
    def __init__(self, parameters, training, item_cats):
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

        # prime the categories' latent factors
        unique_cats = []
        for item in item_cats:
            cats = item_cats[item]
            for cat in cats:
                if cat not in unique_cats:
                    unique_cats.append(cat)
#        for cat in unique_cats:
#            self.cats_latent_factors[cat] = np.zeros(self.f)

        # create a matrix version of this
        self.cats_latent_factors_matrix = np.ndarray(shape=(len(unique_cats), self.f), dtype = float)
        self.prior_cats_latent_factors_matrix = np.ndarray(shape=(len(unique_cats), self.f), dtype = float)

        # prime the average category rating
        for item in item_cats:
            item_cat_avg_rating_vector = np.zeros(len(unique_cats))
            cats = item_cats[item]

            # get the ratings of this item
            item_reviews = [review for review in training.reveiews if review.itemid is item]

            # set the average category rating for each vector element
            avg_rating = sum([review.rating_score for review in item_reviews]) / len(item_reviews)
            for cat in cats:
                item_cat_avg_rating_vector[unique_cats.index(cat)] = avg_rating

        self.epochs = 0

    def reset_hyperparameters(self, hypers):
        self.eta = hypers['eta']
        self.lambd = hypers['lambda']

        # returns the predicted rating of the review
    def apply(self, review):
        predicted_rating = self.mu

        # if we have the user and item bias then use them
        if review.itemid in self.item_biases and review.userid in self.user_biases:
            item_bias = self.item_biases[review.itemid]
            user_bias = self.user_biases[review.userid]

            user_latent_factors = self.user_latent_factors[review.userid]
            item_categories_ratings = self.item_cats_ratings[review.itemid]

            # compute the dot product between the user latent factor vector and the categories by latent factors matrix
            pc_1 = np.dot(user_latent_factors, self.cats_latent_factors_matrix)
            pc_2 = np.dot(pc_1, item_categories_ratings)

            # determine the predicted rating
            predicted_rating += item_bias + user_bias + pc_2

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

        return predicted_rating

    def update(self, review, error):

        if review.itemid in self.item_biases and review.userid in self.user_biases:
            # update the biases
            item_bias = self.item_biases[review.itemid]
            self.item_biases[review.itemid] = item_bias + self.eta * (error - self.lambd * item_bias)

            user_bias = self.user_biases[review.userid]
            self.user_biases[review.userid] = user_bias + self.eta * (error - self.lambd * user_bias)

            # update the latent factor vectors
            user_latent_factors = self.user_latent_factors[review.userid]
            old_user_latent_factors = user_latent_factors


            item_latent_factors += self.eta * (error * old_user_latent_factors - self.lambd * old_item_latent_factors)
            user_latent_factors += self.eta * (error * old_item_latent_factors - self.lambd * old_user_latent_factors)

            self.item_latent_factors[review.itemid] = item_latent_factors
            self.user_latent_factors[review.userid] = user_latent_factors

        # log the error
        self.errors.append(error)


    def write_diagnostics(self):
        # f = open('myfile', 'w')
        # for error in self.errors:
        #     f.write(str(error) + '\n')
        # f.close()
        # print str(self.errors)
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