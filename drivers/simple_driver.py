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

from models.simple_network import SimpleNetwork
from models.rnn import RNN
from drivers.my_driver import MyDriver



class SimpleDriver(MyDriver):

    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """

        command = Command()

        state = self.read_state(carstate)

        try:
            command.steering = self.model(state).data[0] * 0.35
        except:
            print(sys.exc_info())
            os.system('pkill torcs')
            sys.exit()


        print("Steering:", command.steering)

        # ACC_LATERAL_MAX = 6400 * 5
        # v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
        v_x = 180

        self.accelerate(carstate, v_x, command)

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command
