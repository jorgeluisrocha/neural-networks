"""
AUTHOR: jorgeluisrocha
LAST UPDATED: 10-22-2016

This file in its current conception is to create a Perceptron object and
has two functions which allows it to learn how to behave like an OR gate and
an AND gate.
"""

from random import choice
from numpy import array, dot, random
from pylab import plot, ylim

class Perceptron:
    def __init__(self):
    	## Create random weights for all inputs.
        self.w      = random.rand(3)

        self.errors = []    ## Contain lists of errors
        self.eta    = 0.2   ## Learning rate
        self.n      = 100   ## Number of iterations

    def ORGate(self, A, B):
        ## Create a lambda function for the step function in a perceptron.
        unit_step = lambda x: 0 if x < 0 else 1


        ## Here is the training data for what an OR gate represents.
        training_data = [
            (array([0,0,1]), 0),
            (array([0,1,1]), 1),
            (array([1,0,1]), 1),
            (array([1,1,1]), 1),
            ]


        for i in range(self.n):
        	x, expected     = choice(training_data)
        	result          = dot(self.w, x)
        	error           = expected - unit_step(result)
        	self.errors.append(error)
        	self.w         += self.eta * error * x

        data = array([A, B, 1])

        result          = dot(data, self.w)
        print("{}: {} -> {}".format(data[:2], result, unit_step(result)))

    def ANDGate(self, A, B):
    	## Create a lambda function for the step function in a perceptron.
        unit_step = lambda x: 0 if x < 0 else 1


        ## Here is the training data for what an AND gate represents.
        training_data = [
            (array([0,0,1]), 0),
            (array([0,1,1]), 0),
            (array([1,0,1]), 0),
            (array([1,1,1]), 1),
            ]


        for i in range(self.n):
        	x, expected     = choice(training_data)
        	result          = dot(self.w, x)
        	error           = expected - unit_step(result)
        	self.errors.append(error)
        	self.w         += self.eta * error * x

        data = array([A, B, 1])
        
        result          = dot(data, self.w)
        print("{}: {} -> {}".format(data[:2], result, unit_step(result)))


