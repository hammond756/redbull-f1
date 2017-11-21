#! /usr/bin/env python3

import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable as var


def load_data(track):

    # make sure variable 'track' will be usable in the near future..
    data = pd.read_csv('training_data/train_data/aalborg.csv')

    ## x_data being the sensor input
    x_data = data.iloc[:, 3:25].values
    x_dim = x_data.shape[1]

    ## y_data being the steering output (so far)
    y_data = data['STEERING'].values

    print(y_data[:10])

    # Cannot use above method, creating our own:
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.2, random_state=0)

    return x_train, x_valid, y_train, y_valid, x_dim + 1

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
              nn.ReLU(),
              nn.Conv1d(hidden_2, 5),
              nn.ReLu()
            )

    return model

def train_model():

    # TODO
    return

x_train, x_valid, y_train, y_valid, x_dim = load_data("Nothing here yet")

x = var(torch.from_numpy(x_train).type(torch.FloatTensor))
y = var(torch.from_numpy(y_train).type(torch.FloatTensor), requires_grad=False)

print(x)
print(y)

#x = var(torch.randn(len(x_train), x_dim-1))
#y = var(torch.randn(len(x_train), 1), requires_grad=False)


model = torch.nn.Sequential(
    torch.nn.Linear(x_dim - 1, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 1),
)

loss_fn = torch.nn.MSELoss(size_average=False)


learning_rate = 1e-4
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Variable of input data to the Module and it produces
    # a Variable of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Variables containing the predicted and true
    # values of y, and the loss function returns a Variable containing the
    # loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.data[0])

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Variables with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Variable, so
    # we can access its data and gradients like we did before.
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data





#
