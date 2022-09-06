import numpy as np

class BaseLayer:
    def __init__(self, testing_phase=False):
        self.testing_phase = testing_phase
