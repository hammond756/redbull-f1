from pytocl.driver import Driver
from pytocl.car import State, Command
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch

from torch.autograd import Variable as var
import os
import sys

MPS_PER_KMH = 1000 / 3600

class MyDriver(Driver):
    """ Base class for custom drivers """

    model = None

    def __init__(self, model, parameters, logdata=False):
        super().__init__(logdata)
        model.load_state_dict(torch.load(parameters))
        self.model = model

    def read_state(self, carstate):
        """ Transforms carstate as described in car.py to format used to
            train on. """
        parsed_state = []
        parsed_state.append(carstate.speed_x)
        parsed_state.append(carstate.distance_from_center)
        parsed_state.append(carstate.angle)
        parsed_state.extend(list(carstate.distances_from_edge))
        parsed_state = np.array(parsed_state)

        return var(torch.from_numpy(parsed_state).type(torch.FloatTensor))
