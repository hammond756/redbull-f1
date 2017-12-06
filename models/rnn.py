import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable as var
from torch import optim

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math, random, sys, os


class RNN(nn.Module):
    def __init__(self, input_size, hidden_units, output_size):
        super(RNN, self).__init__()

        self.hidden_units = hidden_units
        self.in_dim = input_size
        self.out_dim = output_size

        # define layers
        self.input = nn.Linear(input_size, hidden_units)
        self.rnn = nn.LSTM(hidden_units, hidden_units, num_layers=2, dropout=0.05)
        self.output = nn.Linear(hidden_units, output_size)

    def forward(self, input, hidden):
        inp = self.input(input.view(1,-1)).unsqueeze(1)
        #print("INPUT: ", inp)
        output, hidden = self.rnn(inp, hidden)
        output = self.output(output.view(1, -1))

        return output, hidden

    def init_hidden(self, batch_size):
        return (var(torch.zeros(2, batch_size, self.hidden_units)),
                    var(torch.zeros(2, batch_size, self.hidden_units)))

    def get_n_units(self):
        return (self.in_dim, self.hidden_units, self.out_dim)

# This class is intended for braking and accelerating only
class GASSEN(nn.Module):
    def __init__(self, input_size, hidden_units, output_size):
        super(GASSEN, self).__init__()

        self.hidden_units = hidden_units
        self.in_dim = input_size
        self.out_dim = output_size

        # define layers
        self.input = nn.Linear(input_size, hidden_units)
        self.rnn = nn.LSTM(hidden_units, hidden_units, num_layers=2, dropout=0.05)
        self.output = nn.Linear(hidden_units, output_size)


    def forward(self, input, hidden):
        inp = self.input(input.view(1,-1)).unsqueeze(1)
        #print("INPUT: ", inp)
        output, hidden = self.rnn(inp, hidden)
        output = self.output(output.view(1, -1))

        return F.sigmoid(output), hidden

    def init_hidden(self, batch_size):
        return (var(torch.zeros(2, batch_size, self.hidden_units)),
                    var(torch.zeros(2, batch_size, self.hidden_units)))

    def get_n_units(self):
        return (self.in_dim, self.hidden_units, self.out_dim)

class REMMEN(nn.Module):
    def __init__(self, input_size, hidden_units, output_size):
        super(REMMEN, self).__init__()

        self.hidden_units = hidden_units
        self.in_dim = input_size
        self.out_dim = output_size

        # define layers
        self.input = nn.Linear(input_size, hidden_units)
        self.rnn = nn.LSTM(hidden_units, hidden_units, num_layers=2, dropout=0.05)
        self.output = nn.Linear(hidden_units, output_size)


    def forward(self, input, hidden):
        inp = self.input(input.view(1,-1)).unsqueeze(1)
        #print("INPUT: ", inp)
        output, hidden = self.rnn(inp, hidden)
        output = self.output(output.view(1, -1))

        return output, hidden

    def init_hidden(self, batch_size):
        return (var(torch.zeros(2, batch_size, self.hidden_units)),
                    var(torch.zeros(2, batch_size, self.hidden_units)))

    def get_n_units(self):
        return (self.in_dim, self.hidden_units, self.out_dim)


def train_rnn(test_load=False):

    ####AMEND THESE PARAMETERS:#################################################
    n_epochs = 10
    hidden_units = 17

    trained_name = 'remmen-sigmoid-6tracks-10epochs-10laps'

    csv_files = ['road_aalborg_1cars_2912334.csv',
                 'road_ruudskogen_1cars_28195153.csv',
                 'road_wheel-2_1cars_29114320.csv',
                 'road_forza_1cars_2895946.csv',
                 'road_g-track-1_1cars_28114651.csv',
                 'road_g-track-2_1cars_28192630.csv']
    ###########################################################################

    data = TorcsDataLoader.TorcsTrackDataset(csv_files)
    nr_records = len(data.targets)

    # Use this method to view the loaded data without going straight to training.

    print("First 10 records for states:")
    print(data.carstates[:10])
    print(data.carstates.shape)
    print("Shape: ", data.x_dim)
    print("First 10 records for targets:")
    print(data.targets[:10])
    print(data.targets.shape)
    print("Shape: ", data.t_dim)


    # SELECT MODEL HERE (SIGMOID OR NOT):
    model = REMMEN(data.x_dim, hidden_units, data.t_dim)
    print("Object type: ", model)

    if test_load == True:
        return

    stats = open(trained_name + ".txt", 'w')

    stats.write("Number of epochs: %d\r\n" % (n_epochs))
    stats.write("Hidden_units:     %d\r\n" % (hidden_units))
    stats.write("Files used:\r\n")
    for csv_file in csv_files:
        stats.write("\t - %s\r\n" % csv_file)
    stats.write("Total records: {}\r\n".format(nr_records))


    loss_func = nn.MSELoss()
    optimzer = optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(n_epochs):
        stats.write("Training stats: \r\n")
        total_loss = 0
        for i, observation in enumerate(data):

            x_t = observation['input']
            targets = observation['target']

            if i % 1000 == 0: print("Iteration: ", i)

            hidden = model.init_hidden(1)


            inp = var(torch.FloatTensor(x_t))
            targets = var(torch.FloatTensor([targets]), requires_grad=False)


            outputs, hidden = model(inp, hidden)
            model.zero_grad()

            loss = loss_func(outputs.view(1, -1), targets)
            loss.backward()
            optimzer.step()

            total_loss += loss.data[0]

        # Logging data:
        stats.write("Epoch: {}/{}, Total loss: {}".format(epoch+1, n_epochs, total_loss))
        print('Finishing epoch', epoch+1, 'of', n_epochs, " | Loss", total_loss)

    TorcsDataLoader.save_parameters(model, trained_name)
    stats.close()

if __name__ == '__main__':
    import TorcsDataLoader
    train_rnn(test_load=False)
