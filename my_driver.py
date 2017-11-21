from pytocl.driver import Driver
from pytocl.car import State, Command
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable as var
import my_model


class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    ...
    # def drive(self, carstate: State) -> Command:
    #     # Interesting stuff
    #     command = Command(...)
    #     return command

    def load_data(track):

        # make sure variable 'track' will be usable in the near future..
        data = pd.read_csv('training_data/train_data/aalborg.csv')

        ## x_data being the sensor input
        x_data = data.iloc[:, 3:25].values

        print(len(x_data))
        ## y_data being the steering output (so far)
        y_data = ['STEERING'].values

        x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.2, random_state=0)

        return x_train, x_valid, y_train, y_valid

    def build_model(sen_in, car_in):

        inputs = sen_in + car_in

        hidden_1 = 20
        hidden_2 = 40
        hidden_3 = 50

        output = 1

        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv1d(inputs,hidden_1, 5),
                  nn.ReLU(),
                  nn.Conv1d(hidden_1,hidden_2, 5),
                  nn.ReLU()
                  nn.Conv1d(hidden_2, 5),
                  nn.ReLu()
                )

        return model

    def train_model():

        # TODO

        return model

#
