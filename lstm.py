
""" create an object of lstm model definition

"""
import tensorflow as tf

class LstmModel:

    def __init__(self, units, num_layers):
        self.units = units
        self.layers = num_layers


