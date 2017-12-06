from pytocl.driver import Driver
from pytocl.car import State, Command
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch

from torch.autograd import Variable as var
import os
import sys

path_to_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path_to_project_root)

from models.rnn import RNN
from drivers.my_driver import MyDriver

MPS_PER_KMH = 1000 / 3600
MAX_SPEED = 300
MIN_SPEED = 120


class RNNDriver(MyDriver):

    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """

        command = Command()

        state = self.read_state(carstate)

        current_speed = carstate.speed_x / MPS_PER_KMH

        brk_factor = (current_speed / MAX_SPEED) * 50

        # Braking and accelerating control:\

        max_range = max(carstate.distances_from_edge[8:11])

        try:
            hidden_a = self.model_a.init_hidden(1)
            acc_value = self.model_a(state, hidden_a)[0].data[0][0]
            brk_value = self.model_a(state, hidden_a)[0].data[0][1]

            if acc_value > 0.99:
                if current_speed < MAX_SPEED:
                    if current_speed > MIN_SPEED:
                        if max_range < 150:
                            command.accelerator = 0.0
                            command.brake = brk_value * brk_factor
                        else:
                            command.accelerator = acc_value
                            command.brake = 0.0
                    else:
                        command.accelerator = acc_value
                        command.brake = brk_value
                else:
                    command.accelerator = 0.0

        except:
            print("Using a single model")


        print("Accel value: ", command.accelerator)
        print("Brake value: ", command.brake)
        print("Distance max: ", max_range)

        ## Steering instructions:
        hidden = self.model.init_hidden(1)
        out_steer = self.model(state, hidden)[0].data[0][0]
        print("Carstate angle: ", carstate.angle)
        print("Carstate pos_t: ", carstate.distance_from_center)

        str_factor = 2
        cor_steer = out_steer * str_factor

        command.steering = out_steer

        if carstate.distance_from_center > 0.0:
            if carstate.distances_from_edge[0] > 2:
                if carstate.angle < 0.0:
                    print("USING CORRECTED STEERING")
                    command.steering = cor_steer
        else:
            if carstate.distances_from_edge[18] > 2:
                if carstate.angle > 0.0:
                    print("USING CORRECTED STEERING")
                    command.steering = cor_steer

        print("Steering:", cor_steer)

        print("")
        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1

        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command


    def crashed(edges, track_pos, track_ang, damage):
        if edges[9] < 0.0:
            print(track_pos)
            return True

        return False
