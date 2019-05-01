
import numpy as np 


class Element(object):

    def __init__(self):
        self.ndof = 0
        self.nn = 0
        self.ng = 0
        self.element_type = 0
        self.coord_position = np.array([])

        self.setup()

    def setup(self):
        pass

    def shape_function_value(self):
        pass

    def shape_function_partial(self):
        pass

