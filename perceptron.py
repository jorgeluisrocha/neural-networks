"""
AUTHOR: jorgeluisrocha
LAST UPDATED: 10-21-2016

This perceptron in its current conception is to learn how to utilize an OR gate.
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

    def ORGate(self):
        ## Create an lambda function for the step function in a perceptron.
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

        for x, _ in training_data:
        	result          = dot(x, self.w)
        	print("{}: {} -> {}".format(x[:2], result, unit_step(result)))

    def ANDGate(self):
    	## Create an lambda function for the step function in a perceptron.
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

        for x, _ in training_data:
        	result          = dot(x, self.w)
        	print("{}: {} -> {}".format(x[:2], result, unit_step(result)))


