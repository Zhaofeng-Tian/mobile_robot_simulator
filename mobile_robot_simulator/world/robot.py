import math
from math import atan2, cos, sin, tan, pi
import sys
sys.path.append('C:\\Users\\61602\\Desktop\\Coding\\python')
from mobile_robot_simulator.world.world import World

import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import random
from skimage.draw import disk, polygon

class Robot:
    def __init__(self, world, state = [0., 0., 0., 0., 0., 0], dt = 1, type = "differential", a_limits = [[-0.8,0.2],[-2.6,2.6]], v_limits = [[-1.0,1.0],[-3.0, 3.0]]):
        self. world = world             # initialize the world with the world type, e.g., lawn or maze.
        self. state = state              # x, y, theta (yaw angle), v, w
        self. pre_state = state
        self. control_signal = [0., 0.] # u ,w linear and angular speed
        self. excecut_signal = [0., 0.] # real excecuted signal within acceleration limits
        self. dt = dt                   # control time interval
        self. robot_drive_type = type   # "differential" or "ackermann"
        self. a_limits = a_limits       # linear and rotation acceleration limit
        self. v_limits = v_limits       # linear and rotation speed limit
        self. shape = [0.8, 0.6, (0,0), 0.25]  # lenth, width, deck center offset w.r.t. body center, deck radius
        self. start_pose = None
        self. vertices = None
        self. point_out_map = False
        self. init_pose()
        self. map_copy = self.world.map.copy()
        self. mowing_map = self.world.map.copy()
        
        self. cut_step = 0

    def init_pose(self):
        """ Initialize the pose of the robot on the map,
            And make sure the initialized robot would not 
            collide with any obstacles at the start pose.
        """
        print(" Init robot pose ......................")
        it = 0
        while True:
            it += 1
            print(" Pose initialization interation times: " + str(it))
            if it > 20:
                print (" Cannot find a good pose to init!!!")
                break
            a = self.world.boundary_world_width
            b = self.world.size_x - self.world.boundary_world_width
            bb = self.world.size_y - self.world.boundary_world_width
            sx = random.uniform(a, b)
            sy = random.uniform(a, bb)
            sth = random.uniform(-math.pi, math.pi) # start theta
            if self.body_collision_check(sx,sy,sth) == True:
                continue
            elif self.body_collision_check(sx, sy, sth) == False:
                self.start_pose = [sx, sy, sth]
                self.state[:3] = self.start_pose
                self.state[3:] = [0,0,0]
                self.pre_state = self.state
                print ("start pose initialized: " + str(self.start_pose))
                break

    def get_four_lines(self, x, y, th):
        """(x1,y1), (x2,y2),(x3,y3),(x4,y4)"""
        p1, p2, p3, p4 = self.calc_rect_vertices(x, y, th)
        # print("points to build lines: ", [p1,p2,p3,p4])
        l1 = self.build_line(p1,p2)
        l2 = self.build_line(p2,p3)
        l3 = self.build_line(p3,p4)
        l4 = self.build_line(p4,p1)
        # print("get four lines:",[l1,l2,l3,l4])
        return [l1,l2,l3,l4]

    def line_collision_check(self, line):
        if line == []:
            # print(" Line is empty!")
            return True
        for point in line:
            ir = point[0]
            ic = point[1]
            if ir >= self.world.map_size_y or ic >= self.world.map_size_x:
                return True
            if self. check_point_occupancy(ir, ic) == True:
                # print ("Line collision Detected !")
                return True
        return False
    
    def body_collision_check(self, x,y,th):
        v_lines = self.get_four_lines(x,y,th)
        # print("V_lines: ", v_lines)
        for line in v_lines:
            # print(line)
            if self.line_collision_check(line) == True:
                print ("Collision Detected !")
                return True
        # print (" No collision detected :)")
        return False


    
    def check_point_occupancy(self, ir, ic):
        if self.world.map[ir][ic] == 1:
            return True
        return False
    
    
    def build_line(self, start, end):
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


    
    def calc_rect_vertices(self, x, y, th):
        """ First calculated four vertices' coordinates (x, y),
            then coverted them into indices on the map (ir, ic),
            where row index = y /resolution,
                  col index = x /resolution.
        """
        x = x
        y = y
        theta = th
        b = math.sqrt(self.shape[0]**2 + self.shape[1]**2)/2
        temp = atan2(self.shape[1],self.shape[0])
        #print(temp)

        ph1= theta + temp
        ph2 = theta + pi-temp
        ph3 = theta + pi + temp
        ph4 = theta + 2*pi-temp
        #print([ph1,ph2,ph3,ph4])
        ph1 = atan2(sin(ph1),cos(ph1))
        ph2 = atan2(sin(ph2),cos(ph2))
        ph3 = atan2(sin(ph3),cos(ph3))
        ph4 = atan2(sin(ph4),cos(ph4))
        #print([ph1,ph2,ph3,ph4])

        x1 = x + cos(ph1)*b
        y1 = y + sin(ph1)*b
        x2 = x + cos(ph2)*b
        y2 = y + sin(ph2)*b
        x3 = x + cos(ph3)*b
        y3 = y + sin(ph3)*b
        x4 = x + cos(ph4)*b
        y4 = y + sin(ph4)*b
        self.point_out_map = False
        for i in [x1,y1,x2,y2,x3,y3,x4,y4]:
            if i <= 0.0 or i>= 10.0:
                print(" Point is out, in calc_rect_vertices check!")
                self.point_out_map = True
        r1 = round(y1/self.world.resolution)
        c1 = round(x1/self.world.resolution)
        r2 = round(y2/self.world.resolution)
        c2 = round(x2/self.world.resolution)
        r3 = round(y3/self.world.resolution)
        c3 = round(x3/self.world.resolution)
        r4 = round(y4/self.world.resolution)
        c4 = round(x4/self.world.resolution)
        # print("Got rectangular vertices: ") # indices / not real coords
        # print([(r1,c1), (r2,c2),(r3,c3),(r4,c4)])
        return [(r1,c1), (r2,c2),(r3,c3),(r4,c4)]
                                                                                                                                       
    def print_robot_info(self):
        self.print_shape()
        self.print_state()
        

    def move_base(self, u):
        self. control_signal = [u[0], u[1]]
    
    def clip_control_signal(self, state, d_t):
        """ 
        Clif the control signal so that the real executed 
        signal is within acceleration limits.
        """
        s = state; dt = d_t; clipped = [0,0]
        if self.control_signal[0]-s[3] >= 0: # Throttle
            clipped[0] = min(self.control_signal[0]-s[3], self.a_limits[0][1] * dt, self.v_limits[0][1]-s[3]) 
        else:    # Brake
            clipped[0] = max(self.control_signal[0]-s[3], self.a_limits[0][0] * dt, self.v_limits[0][0]-s[3]) 

        if self.control_signal[1]-s[4] >= 0: # rotate faster
            clipped[1] = min(self.control_signal[1]-s[4], self.a_limits[1][1] * dt, self.v_limits[1][1]-s[4]) 
        else:    # Brake
            clipped[1] = max(self.control_signal[1]-s[4], self.a_limits[1][0] * dt, self.v_limits[1][0]-s[4])
        return clipped # clipped = [delta v, delta w]
    

    

    # def calc_state(self, state, d_t):
    #     s = state ;ss = state; dt = d_t
    #     dv = self.clip_control_signal(s, dt)
    #     ss[3] = s[3] + dv[0]
    #     ss[4] = s[4] + dv[1]
    #     ss[0] = s[0] + math.cos(s[2])* ss[3] * dt  # update x with old speed s[3]
    #     ss[1] = s[1] + math.sin(s[2])* ss[3]  * dt  # update y with old speed
    #     ss[2] = s[2] + ss[4] * dt                           # update yaw angle theta
    #     ss[2] = math.atan2(math.sin(ss[2]), math.cos(ss[2]))# bound [-pi, pi]
    #     ss[5] = self.body_collision_check(ss[0], ss[1], ss[2])
    #     if ss[5] == True:
    #         ss[0] = s[0]; ss[1]=s[1];ss[2] = s[2]
    #     return ss
    
    def calc_state(self, state, d_t):
        s = state.copy() ;ss = state.copy(); dt = d_t
        ss[0] = s[0] + math.cos(s[2])* s[3] * dt  # update x with old speed s[3]
        ss[1] = s[1] + math.sin(s[2])* s[3]  * dt  # update y with old speed
        ss[2] = s[2] + s[4] * dt                           # update yaw angle theta
        ss[2] = math.atan2(math.sin(ss[2]), math.cos(ss[2]))# bound [-pi, pi]
        dv = self.clip_control_signal(s, dt)
        ss[3] = s[3] + dv[0]
        ss[4] = s[4] + dv[1]
        ss[5] = self.body_collision_check(ss[0], ss[1], ss[2])
        return ss

    def state_update(self, visualize = True, draw = True):
        """
        Udate: 
        state, world map, mowing map
        """

        print (" ***************** Update State ******************")
        s = self.state.copy(); dt = self.dt
        # print("s: ",s)
        # print("state: ", self.state)
        ss = self.calc_state(s, dt)
        # print("s: ", s)
        # print("ss: ", ss)
        # print("state: ", self.state)
        if ss[5]:
            # print(" ss[5] = True")
            # print("s: ", s)
            # print("ss: ", ss)
            # print("state: ", self.state)
            self.state[3:] = [0,0,1]
        else:
            # print("state = ss")
            # print("s: ", s)
            # print("ss: ", ss)
            # print("state: ", self.state)
            self.state = ss
        # print("state: ", self.state)
        self.vertices = self.calc_rect_vertices(self.state[0],self.state[1],self.state[2])
        self.cut_step = self.calc_cut_step(self.world.map, self.state.copy())
        # print(" Cut {:} this step. ".format(self.cut_step)) 

        if draw:
            self.draw_mowing_map()
        # visulization option
        # if visualize:
        #     plt.clf()
        #     plt.imshow(self.mowing_map, origin='lower', cmap='gray')
        #     plt.arrow(round(self.state[0]/self.world.resolution),
        #             round(self.state[1]/self.world.resolution),
        #             math.cos(self.state[2])/self.world.resolution,
        #             math.sin(self.state[2])/self.world.resolution,
        #             width = 0.1 /self.world.resolution,
        #             color = 'r')
        #     plt.draw()
        #     plt.pause(0.02)


    def get_local_map(self, size):
        x,y = self.state[0], self.state[1]
        map_size = size
        map = self.mowing_map.copy()
        r = round(y/0.01); c = round(x/0.01)
        rrange =[0,0]; crange=[0,0]
        if r-map_size/2 <= 0: 
            rrange = [0, map_size]
        elif r+map_size/2 >= 1000:
            rrange = [1000-map_size, 1000]
        else:
            rrange = [r-map_size/2, r+map_size/2]
        
        if c-map_size/2 <= 0: 
            crange = [0, map_size]
        elif c+map_size/2 >= 1000:
            crange = [1000-map_size, 1000]
        else:
            crange = [c-map_size/2, c+map_size/2]
        rrange = np.array(rrange, dtype=np.int32)
        crange = np.array(crange, dtype=np.int32)

        # 
        return map[rrange[0]:rrange[1],crange[0]:crange[1]]

    def draw_mowing_map(self):
        self.draw_mowing_disk(self.world.map)
        self.mowing_map = self.world.map.copy()
        self.draw_rect_body(self.mowing_map)  
        self.draw_mowing_disk(self.mowing_map)        


    def fill_rect_body(self, pts, canvas):            
        """ fill robot body rectangular """
        rr = []
        cc = []
        # p (x, y)
        for p in pts:
            rr.append(p[0])
            cc.append(p[1])
        # rr = np.array(rr)
        # cc = np.array(cc)
        canvas[polygon(rr,cc)] = 1
    
    
    def draw_rect_body(self, map):
        """ Draw body using self.state and map"""
        #print(" Drawing rect body.................... ")
        rr = []
        cc = []        
        for p in self.vertices:
            rr.append(p[0])
            cc.append(p[1])
        # rr = np.array(rr)
        # cc = np.array(cc)
        map[polygon(rr,cc)] = 0.75

    def draw_mowing_disk(self, map):
        """ Change the cut area values """
        #print (" Drawing mowing body .................. ")
        row = round(self.state[1]/self.world.resolution)
        col = round(self.state[0]/self.world.resolution)
        r = round(self.shape[3]/self.world.resolution)
        # print(row,col,r)
        map[disk((row,col),r)] = 0.5


    def reset(self):
        """ Reset the robot pose and mowing areas"""
        self.world.map = self.map_copy.copy()
        self.mowing_map = self.map_copy.copy()
        self.init_pose()
        # self.state[:3]= self.start_pose 
        # self.state[3:] = [0,0,0]

    def reset_world(self):
        """ Reset the whole world including map """
        self.world.generate_map() # map reset here
        self.mowing_map = self.world.map.copy()
        self.init_pose() # random a start pose, state reset also
    
    def calc_uncut(self):
        u = np.sum(self.world.map == 0)
        return u
    
    def calc_cut(self):
        c = np.sum(self.world.map == 0.5)
        return c
    
    def calc_cut_step(self,map,state):
        row = round(state[1]/self.world.resolution)
        col = round(state[0]/self.world.resolution)
        r = round(self.shape[3]/self.world.resolution)
        d = disk((row,col),r)
        d0 = np.clip(d[0], 0, 1000)
        d1 = np.clip(d[1], 0, 1000)
        # print(d0)
        # print(d1)
        # print(d)
        uc = np.sum(map[d0,d1]== 0 )
        return uc

    def _get_state(self):
        state = self.state
        return state
    def print_state(self):
        print (" State: " + str(self.state))
    def print_shape(self):
        print("Shape: " + str(self.shape))
        
    

            





