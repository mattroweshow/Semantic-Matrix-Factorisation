__author__ = 'rowem'

from abc import ABCMeta, abstractmethod

class baseSVD:
    __metaclass__ = ABCMeta
    @abstractmethod
    def apply(self, review):
        pass

    @abstractmethod
    def update(self, review, error):
        pass

    @abstractmethod
    def reset_hyperparameters(self, hypers):
        pass

    @abstractmethod
    def derive_average_rating(self):
        pass

    @abstractmethod
    def convergence_check(self):
        pass

    @abstractmethod
    def write_diagnostics(self):
        pass
