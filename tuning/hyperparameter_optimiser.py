__author__ = 'rowem'

from models.svd_base import baseSVD
from models.svd import SVD
from models.svdplusplus import SVDPlusPlus
from models.items_cats_svd import ItemsCatsSVD
import numpy as np
from scipy.optimize import minimize
from learning.sgd_learner import SGD
from dataset.dataset import Dataset
import os
import csv

class Tuner:
    # Tunes the hyperparameters of the model using n-fold cross-validation
    def nfold_cv_hyperparameter_tuner(self, model_id, model_params, hyper_params, folds, item_categories):

        # Set everything as a global param
        self.model_id = model_id
        self.model_params = model_params
        self.folds = folds
        self.item_categories = item_categories

        # Set the initial guess for the hyperparameters
        # convert the hyperparameters to an ndarry format
        x1 = np.asarray([hyper_params[0]['eta'], hyper_params[0]['lambda']])
        # Run the minimisation routine
        res = minimize(self.tuning_function, x1,
               method='nelder-mead',
               options={'xtol': 1e-8, 'disp': True, 'maxiter': 5})
        print str(res)

        # write the tuned hyperparameters to the log directory
        self.write_tuned_hyperparams(res['x'][0], res['x'][1], res['fun'])


    def write_tuned_hyperparams(self, tuned_eta, tuned_lambda, tuned_error):
        dataset_name = self.folds.folds[0].name
        working_dir = os.path.dirname(os.path.realpath(__file__))
        write_path = working_dir + "/../logs/" + dataset_name + "_tuned_model_" + str(self.model_id) + "_f_" + str(self.model_params['f']) + ".tsv"
        out = open(write_path, 'w')
        out.write(str(tuned_eta) + "\t" + str(tuned_lambda) + "\t" + str(tuned_error) + "\n")
        out.close()


    def tuning_function(self, x):
        # prime the model with the hyperparams
        # set the params based on the input
        hyper_params = {'eta': x[0], 'lambda': x[1]}
        self.model_params.update(hyper_params)
        params = self.model_params

        # prime stochastic learning routine
        sgd = SGD()

        # run SVD
        if self.model_id is 1:
            # prime for 10-fold CV
            rmses = []
            for i in range(0, len(self.folds.folds)):

                # prepare the test portion
                test = self.folds.folds[i]
                # prepare the training portion
                train_reviews = []
                train_users = []
                train_items = []
                train_folds = self.folds.folds[:i] + self.folds.folds[i:]
                for fold in train_folds:
                    for review in fold.reviews:
                        train_reviews.append(review)
                        train_users.append(review.userid)
                        train_items.append(review.itemid)
                train = Dataset(test.name, train_reviews, train_users, train_items)

                # prime the model
                svd = SVD(params, train)
                svd_trained = sgd.train_model(svd, train)
                rmse = sgd.test_model(svd_trained, test)
                rmses.append(rmse)
            print str(x) + " | RMSE = " + str(np.mean(rmses))
            return np.mean(rmses)

        # run SVD++
        elif self.model_id is 2:
            # prime for 10-fold CV
            rmses = []
            for i in range(0, len(self.folds.folds)):

                # prepare the test portion
                test = self.folds.folds[i]
                # prepare the training portion
                train_reviews = []
                train_users = []
                train_items = []
                train_folds = self.folds.folds[:i] + self.folds.folds[i:]
                for fold in train_folds:
                    for review in fold.reviews:
                        train_reviews.append(review)
                        train_users.append(review.userid)
                        train_items.append(review.itemid)
                train = Dataset(test.name, train_reviews, train_users, train_items)

                # prime the model
                svd = SVDPlusPlus(params, train)
                svd_trained = sgd.train_model(svd, train)
                rmse = sgd.test_model(svd_trained, test)
                rmses.append(rmse)
            print str(x) + " | RMSE = " + str(np.mean(rmses))
            return np.mean(rmses)

        # run Item-Cats SVD
        elif self.model_id is 3:
            # prime for 10-fold CV
            rmses = []
            for i in range(0, len(self.folds.folds)):

                # prepare the test portion
                test = self.folds.folds[i]
                # prepare the training portion
                train_reviews = []
                train_users = []
                train_items = []
                train_folds = self.folds.folds[:i] + self.folds.folds[i:]
                for fold in train_folds:
                    for review in fold.reviews:
                        train_reviews.append(review)
                        train_users.append(review.userid)
                        train_items.append(review.itemid)
                train = Dataset(test.name, train_reviews, train_users, train_items)

                # prime the model
                svd = ItemsCatsSVD(params, train, self.item_categories)
                svd_trained = sgd.train_model(svd, train)
                rmse = sgd.test_model(svd_trained, test)
                rmses.append(rmse)
            print str(x) + " | RMSE = " + str(np.mean(rmses))
            return np.mean(rmses)


    def retrieve_tuned_parameters(self, dataset_name, model_id , f):
        working_dir = os.path.dirname(os.path.realpath(__file__))
        read_path = working_dir + "/../logs/" + dataset_name + "_tuned_model_" + str(model_id) + "_f_" + str(f) + ".tsv"
        hyper_params = {}
        with open(read_path, 'rb') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            for row in tsvin:
                hyper_params['eta'] = float(row[0])
                hyper_params['lambda'] = float(row[1])
                hyper_params['tuned_error'] = float(row[2])
        return hyper_params