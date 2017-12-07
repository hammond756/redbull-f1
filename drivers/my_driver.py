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
MAX_SPEED = 5
MAX_ANGLE = 20
MIN_DIST = 1
MAX_COUNT = 100
MAX_BACKWARDS = 100

class MyDriver(Driver):
    """ Base class for custom drivers """

    model = None
    stuck = 0
    standstill = 0
    from_start = 0
    backwards = 0

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

    def is_stuck(self, carstate):
        """Adapted from cpp code provided by berniw TORCS driver tuturial"""



        angle = carstate.angle
        track_pos = carstate.distance_from_center
        speed = carstate.speed_x

        data_says_stuck = abs(angle) > MAX_ANGLE and speed < MAX_SPEED \
            and abs(track_pos) > MIN_DIST

        if data_says_stuck:
            if (self.stuck > MAX_COUNT and carstate.distance_from_center * carstate.angle < 0):
                return True
            else:
                self.stuck += 1
                return False
        else:
            self.stuck = 0
            return False

    def standing_still(self, carstate):
        """Same as is_stuck, but the car faces the other side"""

        # if abs(track_pos > MIN_DIST and speed < MAX_SPEED):
        #     if self.standstill > MAX_COUNT:
        #         return True
        #     else:
        #         self.standstill += 1
        #         return False
        # else:
        #     self.standstill = 0
        #     return False

        angle = carstate.angle
        track_pos = carstate.distance_from_center
        speed = carstate.speed_x

        data_says_stuck = abs(angle) > MAX_ANGLE and speed < MAX_SPEED \
            and abs(track_pos) > MIN_DIST

        if data_says_stuck:
            if (self.standstill > MAX_COUNT and carstate.distance_from_center * carstate.angle > 0):
                return True
            else:
                self.standstill += 1
                return False
        else:
            self.standstill = 0
            return False

    def driving_backwards(self, carstate):

        print("T::", carstate.distance_raced)
        print("T-1::", self.from_start)

        if carstate.distance_raced < self.from_start:
            # the line below gives control back to the main controller sooner,
            # but looks like is better to stay in recovery mode for longer
            # self.from_start = carstate.distance_raced
            if self.backwards > MAX_BACKWARDS:
                return True
            else:
                self.backwards += 1
                return False
        else:
            self.from_start = carstate.distance_raced
            return False
