#!usr/bin/env/ python

import numpy as np


class ActivationReLU:
    """Defining Rectified linear function"""
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)