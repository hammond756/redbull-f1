#! /usr/bin/env python3

import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable as var
import matplotlib.pyplot as plt
import sys
import math

def load_data(track):

    # make sure variable 'track' will be usable in the near future..
    aalborg = pd.read_csv('aalborg.csv', index_col=None, header=0)
    speedway = pd.read_csv('f-speedway.csv', index_col=None, header=0)
    alpine = pd.read_csv('alpine-1.csv', index_col=None, header=0)

    data = pd.concat([aalborg, speedway, alpine])
    # data = aalborg

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

def train_model():
    x_train, x_valid, y_train, y_valid, x_dim = load_data("Nothing here yet")
    print("xtrain", x_train)


    x = var(torch.from_numpy(x_train).type(torch.FloatTensor), requires_grad=True)
    y = var(torch.from_numpy(y_train).type(torch.FloatTensor), requires_grad=False)

    model = MaxVerstappen(x_dim, 13, 1)
    loss_fn = torch.nn.MSELoss(size_average=False)
    learning_rate = 1e-6

    for t in range(1000):
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Variable of input data to the Module and it produces
        # a Variable of output data.
        y_pred = model(x)

        # Compute and print loss. We pass Variables containing the predicted and true
        # values of y, and the loss function returns a Variable containing the
        # loss.
        loss = loss_fn(y_pred, y)
        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Variables with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)

        # Update the weights using gradient descent. Each parameter is a Variable, so
        # we can access its data and gradients like we did before.
        for param in model.parameters():
            param.data -= learning_rate * param.grad.data

        if t % 100 == 0:
            print("epoch", t, "total loss", loss.data)

    data = model(var(torch.FloatTensor(x_valid)))
    data = data.data.numpy()

    # for param in model.parameters():
    #     print(param.data)

    plt.plot(range(len(data)), data)
    plt.show()

    torch.save(model.state_dict(), 'steering-all_given_tracks.h5')
    return

class MaxVerstappen(nn.Module):
    def __init__(self, in_dim, hidden_units, out_dim):
        super(MaxVerstappen, self).__init__()
        self.lin1 = torch.nn.Linear(in_dim, hidden_units)
        self.lin2 = torch.nn.Linear(hidden_units, out_dim)

    def forward(self, inputs):
        out = self.lin1(inputs)
        # out = F.sigmoid(out)
        out = self.lin2(out)
        out = F.tanh(out)
        return out

if __name__ == '__main__':
    train_model()
