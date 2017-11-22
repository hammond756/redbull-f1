from pytocl.driver import Driver
from pytocl.car import State, Command
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch

from redbullf1.my_model import MaxVerstappen
from redbullf1.rnn import DrivingRNN

from torch.autograd import Variable as var
import os
import sys

MPS_PER_KMH = 1000 / 3600

class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    ...
    # def drive(self, carstate: State) -> Command:
    #     # Interesting stuff
    #     command = Command(...)
    #     return command

    model = None

    def __init__(self, logdata=False):
        super().__init__(logdata)
        path = os.path.dirname(os.path.abspath(__file__))
        # model = MaxVerstappen(22, 13, 1)
        model = DrivingRNN(15, 22, 1)
        model.load_state_dict(torch.load(os.path.join('redbullf1', 'steering-aalborg-lstm.h5')))
        # model.load_state_dict(torch.load(os.path.join('redbullf1', 'steering-all_given_tracks.h5')))
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

    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """

        command = Command()

        state = self.read_state(carstate)

        # needed for rnn
        hidden = self.model.init_hidden(1)

        try:
            command.steering = self.model(state, hidden)[0].data[0]
            # command.steering = self.model(state).data[0]
            print(command.steering)
        except:
            print(sys.exc_info())
            os.system('pkill torcs')
            sys.exit()


        print("Steering:", command.steering)

        # ACC_LATERAL_MAX = 6400 * 5
        # v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
        v_x = 80

        self.accelerate(carstate, v_x, command)

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command
