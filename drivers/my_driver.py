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

    def __init__(self, model, parameters, model_a=None, parameters_a=None, model_b=None, parameters_b=None, logdata=False):
        super().__init__(logdata)
        model.load_state_dict(torch.load(parameters))
        self.model = model

        if model_a is not None:
            model_a.load_state_dict(torch.load(parameters_a))
            self.model_a = model_a

        if model_b is not None:
            model_b.load_state_dict(torch.load(parameters_b))
            self.model_b = model_b


    def read_state(self, carstate):
        """ Transforms carstate as described in car.py to format used to
            train on. """
        parsed_state = []
        parsed_state.append(carstate.speed_x)
        parsed_state.append(carstate.distance_from_center)
        parsed_state.append(carstate.angle)

        # Manipulate sensor data:
        edge_list = list(carstate.distances_from_edge)

        ##### Comment/uncomment for manipulated sensorinputs
        #front_sensors = edge_list[8:11]
        #parsed_state.append(max(front_sensors))
        #edges_neg = edge_list[:8]
        #edges_pos = edge_list[11:]
        #edge_list = np.append(edges_neg, edges_pos)

        #print(edge_list)


        parsed_state.extend(edge_list)
        parsed_state = np.array(parsed_state)


        return var(torch.from_numpy(parsed_state).type(torch.FloatTensor))
