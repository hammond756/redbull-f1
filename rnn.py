import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable as var
from torch import optim

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math, random, sys, os

class DrivingRNN(nn.Module):
    def __init__(self, hidden_units, input_size, output_size):
        super(DrivingRNN, self).__init__()

        self.hidden_units = hidden_units
        self.input_size = input_size
        self.output_size = output_size

        # define layers
        self.input = nn.Linear(input_size, hidden_units)
        # num_layers = 2
        self.rnn = nn.LSTM(hidden_units, hidden_units, 2, dropout=0.05)
        self.output = nn.Linear(hidden_units, output_size)

    # # internal version of forward (i think)
    # def step(self, input, hidden=None):
    #     trans_input = self.input(input.view(1, -1)).unsqueeze(1)
    #     out, hidden = self.rnn(trans_input, hidden)
    #     out = self.output(out.squeeze(1))
    #     return out, hidden

    # def forward(self, inputs, hidden=None, force=True, steps=0):
    #     if force or steps == 0:
    #         steps = len(inputs)
    #
    #     # initialize buffer for memory
    #     outputs = var(torch.zeros(steps, 1, 1))
    #
    #     # iterate over lenght of inputs (each inp is 1xinput_size)
    #     for i in range(steps):
    #         if force or i == 0:
    #             inp = inputs[i]
    #         else:
    #             # here you replace the input by the previous output
    #             inp = output
    #
    #         # do regular forward pass based on current input
    #         output, hidden = self.step(inp, hidden)
    #
    #         # add ouput to mempory
    #         outputs[i] = output
    #
    #     return outputs, hidden

    def forward(self, input, hidden):
        inp = self.input(input.view(1,-1)).unsqueeze(1)
        output, hidden = self.rnn(inp, hidden)
        output = self.output(output.view(1, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        return (var(torch.zeros(2, batch_size, self.hidden_units)),
                    var(torch.zeros(2, batch_size, self.hidden_units)))

def load_data(track=None):

    # make sure variable 'track' will be usable in the near future..
    aalborg = pd.read_csv('aalborg.csv', index_col=None, header=0)
    # speedway = pd.read_csv('f-speedway.csv', index_col=None, header=0)
    # alpine = pd.read_csv('alpine-1.csv', index_col=None, header=0)

    # data = pd.concat([aalborg, speedway, alpine])
    data = aalborg

    ## x_data being the sensor input
    x_data = data.iloc[:, 3:].values
    x_dim = x_data.shape[1]
    print('x_dim', x_dim)

    ## y_data being the steering output (so far)
    y_data = data.iloc[:, 2].values

    print(len(y_data))

    # Cannot use above method, creating our own:
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.2, random_state=0)

    assert not np.any([np.any([math.isnan(x_ij) for x_ij in x_i]) for x_i in x_train])

    return x_train, x_valid, y_train, y_valid, x_dim

def train_rnn():
    n_epochs = 5
    hidden_size = 15
    force_rate = 0.5

    x_train, x_valid, y_train, y_valid, input_dim = load_data()

    model = DrivingRNN(hidden_size, input_dim, 1)
    loss_func = nn.MSELoss()
    optimzer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(n_epochs):
        total_loss = 0
        for i, x_t in enumerate(x_train):
            if i % 1000 == 0: print(i)

            hidden = model.init_hidden(1)

            inp = var(torch.FloatTensor(x_t))
            target = var(torch.FloatTensor([y_train[i]]), requires_grad=False)

            outputs, hidden = model(inp, hidden)
            model.zero_grad()

            loss = loss_func(outputs.view(1, -1), target)
            loss.backward()
            optimzer.step()

            total_loss += loss.data[0]
        if epoch % 10 == 0:
            print('Epoch', epoch, "Loss", total_loss)

    torch.save(model.state_dict(), 'steering-aalborg-lstm.h5')

if __name__ == '__main__':
    train_rnn()
