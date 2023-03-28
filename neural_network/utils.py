import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pickle
import matplotlib.ticker as ticker
import math 

L=57
nbvar=4

from PermutationInvariantLayer import *
from symmetries import *
from PermutationInvariantLayer import *

def sign(x):
    return int(math.copysign(1,x))

dir='../gp-mood-code/'

class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class MyDataset(Dataset):
    def __init__(self, data_dict):
        self.inputs = data_dict['inputs']
        self.labels = data_dict['labels']

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

def format(x):
    """Format a double with one digit after the comma"""
    return f"{x:.5f}"


def plot_loss(epoch, lr, tr_loss, te_loss, length, model_name):
    fig, ax1 = plt.subplots()

    # plot the loss on the left y-axis
    ax1.plot(epoch, tr_loss, color='blue', linestyle='dotted', label='Training')
    ax1.plot(epoch, te_loss, color='blue', label='Testing')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Losses', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))
    ax1.legend(loc='upper left')

    # create a second y-axis on the right
    ax2 = ax1.twinx()

    # plot the learning rate on the right y-axis
    ax2.plot(epoch, lr, color='red')
    ax2.set_ylabel('Learning Rate', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_yscale('log')
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))

    plt.title('L = ' + str(length))

    plt.savefig('losses_'+model_name+'_epoch_L_' + str(length) + '.png', dpi=300)
    plt.clf()
    ax1.cla()
    ax2.cla()
    plt.close(fig)



