#!/usr/bin/env/ python

from layer import LayerDense
from data import spiral_data
from activation_function import ActivationReLU
import numpy as np

# Setting seed
np.random.seed(0)

# Dummy data
X, y = spiral_data(100, 3)

layer1 = LayerDense(2, 5)
activation1 = ActivationReLU()

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)
