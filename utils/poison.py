import numpy as np

def make_trigger(x,row=200,col=200,wide=5):
    x[(row):(row+wide),(col):(col+wide)] = 255.0
    return x

def make_trigger_label(y,targeted=0):
    y = targeted # targeted
    return y

