import sys
sys.path.append('C:\\Users\\61602\\Desktop\\Coding\\python')
import numpy as np
from math import pi
class CarParam:
    def __init__(self):
        self.world_resolution = 0.01
        self.kinematic_constraints=np.array([[-1,-0.6],[1,0.6]])
        self.dynamic_constraints = np.array([[-0.8,-1],[0.8,1]])
        # [Lw,Lf,Lr,W] wheelbase,front suspension, rear suspension, width, lidar position to rear axis center
        self.body_structure = np.array([0.53,0.25,0.25,0.68, 0.65])
        self.dt = 0.2
        self.max_range = 6.0 # Laser scan max range
        self.n_scans = 120 # Number of rays
        self.interval = 2*pi/self.n_scans
    
    
