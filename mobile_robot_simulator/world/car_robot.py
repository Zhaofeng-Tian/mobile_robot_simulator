import math
from math import atan2, cos, sin, tan, pi, sqrt
import sys
sys.path.append('C:\\Users\\61602\\Desktop\\Coding\\python')
from mobile_robot_simulator.world.world import World
from mobile_robot_simulator.world.param import CarParam
from mobile_robot_simulator.world.utils import plot_car
from collections import deque
import time

import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import random
from skimage.draw import disk, polygon

class CarRobot:
    def __init__(self, state, param, id):
        # Initiate 
        self.resolution = param.world_resolution
        self.state = state # [x,y,theta,v,phi]
        
        # Global variables
        self.sequence = deque(maxlen=100) # History of states
        self.control_signal = [0,0] # Control signal 
        # self.execut_signal = [0,0] # Actually executed signal by actuators
        # Constant parameters
        self.dt = param.dt
        self.v_limits = param.kinematic_constraints
        self.a_limits = param.dynamic_constraints
        self.body_structure = param.body_structure
        self.vertices = None
        self.vertice_coords = None
        self.disc_centers = None
        self.thresh = 0.9+0.01*id
        self.n_scans = param.n_scans
        self.interval = param.interval
        self.max_range = param.max_range
        self.obs = None
        self.rays = None

        self.calc_disc_lidar_centers()
        self.calc_rect_vertices()

        self.disc_r = 0.5*sqrt((np.sum(self.body_structure[:4])**2/4 + self.body_structure[3]**2))
    
    def move_base(self, u):
        self. control_signal = [u[0], u[1]]
    def clip_control_signal(self):
        """ 
        Clif the control signal so that the real executed 
        signal is within acceleration limits.
        """
        s = self.state.copy(); dt = self.dt; clipped = [0,0]
        if self.control_signal[0]-s[3] >= 0: # Throttle
            clipped[0] = min(self.control_signal[0]-s[3], self.a_limits[1][0] * dt, self.v_limits[1][0]-s[3]) 
        else:    # Brake
            clipped[0] = max(self.control_signal[0]-s[3], self.a_limits[0][0] * dt, self.v_limits[0][0]-s[3]) 

        if self.control_signal[1]-s[4] >= 0: # steer
            clipped[1] = min(self.control_signal[1]-s[4], self.a_limits[1][1] * dt, self.v_limits[1][1]-s[4]) 
        else:    # Brake
            clipped[1] = max(self.control_signal[1]-s[4], self.a_limits[0][1] * dt, self.v_limits[0][1]-s[4])
        return clipped # clipped = [delta v, delta w]
    
    def state_update(self,map):
        # First store the old state
        old_state = self.state.copy()
        self.sequence.append(old_state)
        # Update actuator state
        clipped = self.clip_control_signal()
        self.state[3:] += clipped
        # Update pose state
        self.state[:3] = old_state[:3] + np.array([cos(old_state[2]),sin(old_state[2]),tan(self.state[4])/self.body_structure[0]])*self.state[3]*self.dt
        self.calc_rect_vertices()
        self.calc_disc_lidar_centers()
        # self.fill_rect_body(self.vertices, map, self.thresh)
        # self.get_scans(map)
    
    def calc_disc_lidar_centers(self):
        x_fd = self.state[0] + (3*self.body_structure[0]+3*self.body_structure[1]-self.body_structure[2])*cos(self.state[2])/4
        y_fd = self.state[1] + (3*self.body_structure[0]+3*self.body_structure[1]-self.body_structure[2])*sin(self.state[2])/4
        x_rd = self.state[0] + (self.body_structure[0]+self.body_structure[1]-3*self.body_structure[2])*cos(self.state[2])/4
        y_rd = self.state[1] + (self.body_structure[0]+self.body_structure[1]-3*self.body_structure[2])*sin(self.state[2])/4
        x_lidar = self.state[0]+self.body_structure[4]*cos(self.state[2])
        y_lidar = self.state[1]+self.body_structure[4]*sin(self.state[2])
        self.disc_centers = [(x_fd,y_fd),(x_rd,y_rd),(x_lidar,y_lidar)]

    def calc_rect_vertices(self):
        """ First calculated four vertices' coordinates (x, y),
            then coverted them into indices on the map (ir, ic),
            where row index = y /resolution,
                  col index = x /resolution.
        """
        x = self.state[0]
        y = self.state[1]
        theta = self.state[2]
        fdiag = sqrt((self.body_structure[0]+self.body_structure[1])**2 
                           + (self.body_structure[3]/2)**2)
        rdiag = sqrt((self.body_structure[2])**2 
                           + (self.body_structure[3]/2)**2)
        temp1 = atan2(self.body_structure[3]/2,self.body_structure[0]+self.body_structure[1])
        temp2 = atan2(self.body_structure[3]/2,self.body_structure[2])
        #print(temp)

        ph1= theta + temp1
        ph2 = theta + pi-temp2
        ph3 = theta + pi + temp2
        ph4 = theta + 2*pi-temp1
        #print([ph1,ph2,ph3,ph4])
        ph1 = atan2(sin(ph1),cos(ph1))
        ph2 = atan2(sin(ph2),cos(ph2))
        ph3 = atan2(sin(ph3),cos(ph3))
        ph4 = atan2(sin(ph4),cos(ph4))
        #print([ph1,ph2,ph3,ph4])

        x1 = x + cos(ph1)*fdiag
        y1 = y + sin(ph1)*fdiag
        x2 = x + cos(ph2)*rdiag
        y2 = y + sin(ph2)*rdiag
        x3 = x + cos(ph3)*rdiag
        y3 = y + sin(ph3)*rdiag
        x4 = x + cos(ph4)*fdiag
        y4 = y + sin(ph4)*fdiag
        self.vertice_coords = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        r1 = round(y1/self.resolution)
        c1 = round(x1/self.resolution)
        r2 = round(y2/self.resolution)
        c2 = round(x2/self.resolution)
        r3 = round(y3/self.resolution)
        c3 = round(x3/self.resolution)
        r4 = round(y4/self.resolution)
        c4 = round(x4/self.resolution)
        # print("Got rectangular vertices: ") # indices / not real coords
        # print([(r1,c1), (r2,c2),(r3,c3),(r4,c4)])
        self.vertices = [(r1,c1), (r2,c2),(r3,c3),(r4,c4)]
        # return [(r1,c1), (r2,c2),(r3,c3),(r4,c4)]

    def get_scans(self,map, noise=True):
        ends = self.build_ray_ends()
        # print("ends: ", ends)
        rays = []
        obs = []
        lidar_cell = self.p2i(self.disc_centers[2],self.resolution)
        # print("**********", lidar_cell)
        for (r,c) in ends:
            ray = self.build_a_ray(lidar_cell,(r,c),map,self.thresh)
            if noise:
                obs.append(self.dist_i2p(lidar_cell,ray[-1])+random.uniform(-0.05,0.05))
            else:
                obs.append(self.dist_i2p(lidar_cell,ray[-1]))
            rays.append(ray)
        self.obs = obs # list
        self.rays = rays
        return rays,obs

    def dist_i2p(self, start, end):
        # index to position, take (r,c) output a range/m
        r1,c1 = start
        r2,c2 = end
        dist = sqrt((r1-r2)**2+(c1-c2)**2)
        return dist*self.resolution

    @staticmethod
    def build_a_ray( start,end,map,thresh):
        r1,c1 = start; r2,c2 = end
        dr = r2-r1; dc = c2-c1
        points = []
        if abs(dr)>abs(dc):
            for i in range(abs(dr)):
                r = r1+ round(i*dr/abs(dr))
                c = c1+ round(i*dc/abs(dr))
                points.append((r,c))
                if map[r][c] >= 0.5 and (map[r][c]>= thresh+0.005 or map[r][c]<=thresh-0.005):
                    # print(" breaking la!!!",r,c, map[r][c])
                    break;
        else:
            for i in range(abs(dc)):
                r = r1+ round(i*dr/abs(dc))
                c = c1+ round(i*dc/abs(dc))
                points.append((r,c))
                if map[r][c] >= 0.5 and (map[r][c]>= thresh+0.005 or map[r][c]<=thresh-0.005):
                    # print(" breaking la!!!",r,c, map[r][c])
                    break;
        return points


    # @staticmethod
    # def build_a_ray( start,end,map,thresh):

    #     # here (x, y) = (row, col)
    #     x1, y1 = start
    #     x2, y2 = end
    #     dx = x2 - x1
    #     dy = y2 - y1
    #     is_steep = abs(dy) > abs(dx)  # determine how steep the line is
    #     if is_steep:  # rotate line
    #         x1, y1 = y1, x1
    #         x2, y2 = y2, x2
    #     # swap start and end points if necessary and store swap state
    #     swapped = False
    #     if x1 > x2:
    #         x1, x2 = x2, x1
    #         y1, y2 = y2, y1
    #         swapped = True
    #     dx = x2 - x1  # recalculate differentials
    #     dy = y2 - y1  # recalculate differentials
    #     error = int(dx / 2.0)  # calculate error
    #     y_step = 1 if y1 < y2 else -1
    #     # iterate over bounding box generating points between start and end
    #     y = y1
    #     points = []
    #     for x in range(x1, x2 + 1):

    #         if is_steep:

    #             coord = [y, x]
    #             points.append(coord)
    #             error -= abs(dy)
    #             if error < 0:
    #                 y += y_step
    #                 error += dx
    #             if map[y][x] >= 0.5 and (map[y][x]>= thresh+0.005 or map[y][x]<=thresh-0.005):
    #                 print(" ~~breaking la!!!",y,x, map[y][x])
    #                 break;
    #         else:

    #             coord = (x, y)
    #             points.append(coord)
    #             error -= abs(dy)
    #             if error < 0:
    #                 y += y_step
    #                 error += dx
    #             if map[x][y] >= 0.5 and (map[x][y]>= thresh+0.005 or map[x][y]<=thresh-0.005):
    #                 print(" breaking la!!!",x,y, map[x][y])
    #                 break;
    #     if swapped:  # reverse the list if the coordinates were swapped
    #         points.reverse()
    #     # print (points)
    #     return points

    def build_ray_ends(self):
        ran = self.max_range
        theta = self.state[2]
        x,y = self.disc_centers[2]
        return [self.p2i((x+ran*cos(theta+self.interval*i),y+ran*sin(theta+self.interval*i)),self.resolution) for i in range(self.n_scans)]

    @staticmethod
    def fill_rect_body( pts, canvas, value):            
        """ fill robot body rectangular """
        rr = []
        cc = []
        # p (x, y)
        for p in pts:
            rr.append(p[0])
            cc.append(p[1])
        # rr = np.array(rr)
        # cc = np.array(cc)
        canvas[polygon(rr,cc)] = value

    @staticmethod
    def build_line(start, end):
        """
        Implementation of Bresenham's line drawing algorithm
        Adapted from python robotics
        """
        # print ("Building line .............")
        # print("start point: ", start)
        # print("end point: ", end)
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        is_steep = abs(dy) > abs(dx)  # determine how steep the line is
        if is_steep:  # rotate line
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        # swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
        dx = x2 - x1  # recalculate differentials
        dy = y2 - y1  # recalculate differentials
        error = int(dx / 2.0)  # calculate error
        y_step = 1 if y1 < y2 else -1
        # iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = [y, x] if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx
        if swapped:  # reverse the list if the coordinates were swapped
            points.reverse()
        # print (points)
        return points
    
    @staticmethod
    def p2i(point,res):
        """ Convert point (x,y) to cell index (r,c)"""
        x,y = point
        r = round(y/res)
        c = round(x/res)
        return (r,c)


# if __name__ == "__main__":
#     param = CarParam()
#     world = World(size_x = 20, size_y = 20, resolution = 0.01)
#     robot = CarRobot(world,np.array([5.,5.,0.,0.,0.]),param)
#     robots = []
#     for i in range(10):
#         robot = CarRobot(world,np.array([5+0.8*i,5+0.8*i,0.,0.,0.]),param)
#         robots.append(robot)

#     for i in range(int(60/robot.dt)):

#         map = world.map.copy()
#         start = time.time()
#         for robot in robots:
#             start_time = time.time()
#             robot.move_base([10,10])
#             print(robot.control_signal)
#             robot.state_update()
#             # print(robot.state)
#             print(robot.vertices)
#             robot.fill_rect_body(robot.vertices,map)
#             end_time = time.time()
#             print("Time costing for one loop: ", end_time-start_time)
#         end = time.time()
#         print("total_time: ", end-start)

#         plt.clf()
#         plt.imshow(map, origin='lower', cmap='gray')
#         plt.draw()
#         plt.pause(0.01)

#     print(len(robot.sequence))
