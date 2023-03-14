import numpy as np
from math import ceil
class World:

    def __init__(self, size_x = 10, size_y = 10, resolution = 0.01):
        """ Initialize the world parameters """
        # size
        self.size_x = size_x # [m]
        self.size_y = size_y # [m]
        self.resolution = resolution # [m]
        self. map_size_x = ceil(self.size_x / self.resolution) + 1 # 10/0.01
        self. map_size_y = ceil(self.size_y / self.resolution) + 1 
        self. map = np.zeros((self.map_size_y, self.map_size_x))
        
    def print_world_info(self):
        print( "world size: " + str(self.size_x) + " x " + str(self.size_y))
        print( "resolution: " + str(self.resolution))
    
