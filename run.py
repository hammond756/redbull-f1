#! /usr/bin/env python3
import os
import time
import sys
from pytocl.main import main

import torch

from models.simple_network import SimpleNetwork
from models.rnn import RNN, GASSEN, REMMEN

from drivers.simple_driver import SimpleDriver
from drivers.rnn_driver import RNNDriver

model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')


def get_int_units(file_name):
    return [int(x) for x in file_name.split('-')]

def parse_model_name(file_name):
    # split into label, n_units and modelname
    label, units, model_name = file_name.split('_')[1:]

    # extract integers from n_units
    units = get_int_units(units)

    return units, model_name[:-3]


if __name__ == '__main__':

    driver = None

    # Change second argument to saved model and execute 'python run.py'
    # the rest is maganged by the script. Couldn't get cmd-line arguments
    # to work because they interfere with pytocl

    # STUUR MODEL HIER
    model_parameters = os.path.join(model_dir, 'sturen-6tracks-30epochs-10laps_22-17-1_RNN.h5')

    # GASSEN MODEL HIER
    model_parameters_a = os.path.join(model_dir, 'gassen-sigact-6tracks-20epochs-10laps_22-17-2_GASSEN.h5')

    # REMMEN MODEL HIER - Deze wordt niet gebruikt!!!!
    # model_parameters_b = os.path.join(model_dir, 'remmen-sigmoid-6tracks-10epochs-10laps_22-17-1_REMMEN.h5')

    units, model_name = parse_model_name(model_parameters)
    units_a, model_name_a = parse_model_name(model_parameters_a)
    # units_b, model_name_b = parse_model_name(model_parameters_b)

    if model_name == 'SimpleNetwork':
        model = SimpleNetwork(units[0], units[1], units[2])
        driver = SimpleDriver(model, model_parameters)
    if model_name == 'RNN':
        model = RNN(units[0], units[1], units[2])
        accel = GASSEN(units_a[0], units_a[1], units_a[2])
        # brake = REMMEN(units_b[0], units_b[1], units_b[2])
        driver = RNNDriver(model, model_parameters,
                           accel, model_parameters_a)#, model_parameters_b)

    print("Model used: ", model_name)
    print("Input dimension:  ", units[0])
    print("Hidden units:     ", units[1])
    print("Output dimension: ", units[2])

    # Automatically start torcs
    os.system('pkill torcs')
    time.sleep(0.5)
    os.system('torcs -nofuel -nolaptime &')

    time.sleep(0.5)

    # Auto naviagate GUI
    os.system('sh autostart.sh')
    time.sleep(0.5)

    # Start client
    main(driver)
