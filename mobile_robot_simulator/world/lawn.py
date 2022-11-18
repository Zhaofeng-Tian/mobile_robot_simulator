import math
import sys
sys.path.append('C:\\Users\\61602\\Desktop\\Coding\\python')
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import random
from skimage.draw import polygon, disk
from mobile_robot_simulator.world.world import World

class Lawn (World):
    def __init__ (self, boundary_flag = 1, boundary_world_width = 1.7, sx = 10, sy = 10, rs = 0.01):
        super(Lawn, self).__init__(size_x = sx, size_y = sy, resolution = rs)
        self. boundary_flag = boundary_flag
        self. map_size_x = math.ceil(self.size_x / self.resolution) + 1 # 10/0.01
        self. map_size_y = math.ceil(self.size_y / self.resolution) + 1 
        self. boundary_world_width = boundary_world_width # boundary area width
        self. obstacle_disk_rmin = 0.2
        self. obstacle_disk_rmax = 1.5
        self. map = None
        self. generate_map()


    def generate_map(self):
        """ Discrete lawn model build up """
        print("Building world, please wait .......")
        self.map_init()
        if self. boundary_flag == 1:
            self.random_boundary()
        print("Successfully built!")
        
    
    def map_init(self):
        self. map = np.ones((self.map_size_y, self.map_size_x))


    def random_boundary(self):
        bx, by = self.interpolate_boundary_points()
        bx , by = self.continuity_process(bx,by)
        self.fill_boundary (bx, by)

    def fill_boundary(self,bx,by):
        # Truncate the lawn to fit the map size
        for i in range(0,len(bx)):
            # Clif y
            if by[i] <0:
                by[i] = 0
            elif by[i] > self.map_size_y -1:
                by[i] = self.map_size_y - 1
            # Clif x
            if bx[i] <0:
                bx[i] = 0
            elif bx[i] > self.map_size_x -1:
                bx[i] = self.map_size_x - 1
        v_ir = by
        v_ic = bx
        self.map[polygon(v_ir,v_ic)]=0

    """
    def fill_boundary(self, bx, by):
        # Truncate the lawn to fit the map size
        for i in range(0,len(bx)):
            # Clif y
            if by[i] <0:
                by[i] = 0
            elif by[i] > self.map_size_y -1:
                by[i] = self.map_size_y - 1
            # Clif x
            if bx[i] <0:
                bx[i] = 0
            elif bx[i] > self.map_size_x -1:
                bx[i] = self.map_size_x - 1
                
            self.map[int(by[i])][int(bx[i])] = 1
        m1 = self.map.copy()
        m2 = self.map.copy() 
        # Fill along the y axis
        for col in range(0, self.map_size_x):
            for row in range(0, self.map_size_y):
                if m2[row][col] != 1:
                    m2[row][col] = 1
                elif m2[row][col] == 1:
                    break;
            for row in range(self.map_size_y-1, -1, -1):
                if m2[row][col] != 1:
                    m2[row][col] = 1
                elif m2[row][col] == 1:
                    break;
        # Fill along the x axis
        for r in range(0, self.map_size_y):
            for c in range(0, self.map_size_x):
                if m1[r][c] != 1:
                    m1[r][c] = 1
                elif m1[r][c] == 1:
                    break;
            for c in range(self.map_size_x-1, -1, -1):
                if m1[r][c] != 1:
                    m1[r][c] = 1
                elif m1[r][c] == 1:
                    break;
        # Combine two results
        for col in range(0, self.map_size_x):
            for row in range(0, self.map_size_y):
                if m1[row][col] == 1:
                    self.map[row][col] = 1
                if m2[row][col] == 1:
                    self.map[row][col] = 1
        return self.map
    """



    def continuity_process(self, x_new, y_new): 
        """ Make boundary of a polygon (can be concave) continuity
            w.r.t. x,y coordinates, by first rounding the floats to 
            ints, then insert x,y coordinates to the arrays.
            For instance, check (100, 200)'s next coordinate pair
            (102, 203), dx = 2, dy = 3.  x +=1  y += floor(dy\dx)
            -> (101, 201)-> (102,202)->(102,203)        
        """ 
        x_int = np.rint(x_new)
        y_int = np.rint(y_new)
        [len] = x_int.shape
        i=0
        # Insert more points to make coordinates continuous
        while True:
            if abs(y_int[i] - y_int[i+1]) > 1 and abs(x_int[i] - x_int[i+1]) > 1:
                if x_int[i+1] - x_int[i] > 0:
                    x_int = np.insert(x_int,i+1, x_int[i]+1)
                else:
                    x_int = np.insert(x_int,i+1, x_int[i]-1)
                if y_int[i+1] - y_int[i] > 0:
                    y_int = np.insert(y_int,i+1, y_int[i]+1)
                else:
                    y_int = np.insert(y_int,i+1, y_int[i]-1)
            elif abs(x_int[i] - x_int[i+1]) > 1 and abs(y_int[i] - y_int[i+1]) <= 1:
                if x_int[i+1] - x_int[i] > 0:
                    x_int = np.insert(x_int,i+1, x_int[i]+1)
                    y_int = np.insert(y_int,i+1, y_int[i])
                else:
                    x_int = np.insert(x_int,i+1, x_int[i]-1)
                    y_int = np.insert(y_int,i+1, y_int[i]) 
            elif abs(x_int[i] - x_int[i+1]) <= 1 and abs(y_int[i] - y_int[i+1]) > 1:
                if y_int[i+1] - y_int[i] > 0:
                    y_int = np.insert(y_int,i+1, y_int[i]+1)
                    x_int = np.insert(x_int,i+1, x_int[i])
                else:
                    y_int = np.insert(y_int,i+1, y_int[i]-1)
                    x_int = np.insert(x_int,i+1, x_int[i])
            i += 1
            if i >= x_int.shape[0]-1:
                break;
        # Remove the adjacent repeated points
        while True:
            if i >= x_int.shape[0]-1:
                break;
            while x_int[i] == x_int[i+1] and y_int[i] == y_int[i+1]:
                x_int = np.delete(x_int, i+1)
                y_int = np.delete(y_int, i+1)
            i += 1
        return x_int, y_int





    
    def interpolate_boundary_points(self):
        """ Spline interpolate the points """
        x, y = self.generate_boundary_points()
        x.append(x[-1])
        y.append(y[-1])
        xx = np.array(x)
        yy = np.array(y)
        pts = np.stack ((xx,yy))
        tck, u = splprep(pts, u=None, s=0.0, per=1) 
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new, y_new = splev(u_new, tck, der=0)
        return x_new,y_new

    def generate_boundary_points(self):
        """^ y
           |---------------------
           |8 |       1       |2 |  
           ------------------------
           |  |               |  |
           |7 |               |3 |
           ------------------------
           |6 |       5       |4 |
           ------------------------> x
        Generate curve boundary points randomly in 8 areas
        """
        boudary_map_width = math.ceil(self.boundary_world_width / self.resolution)
        n_row_points = math.ceil((self.map_size_x - 2* boudary_map_width)/ boudary_map_width)
        n_clm_points = math.ceil((self.map_size_y - 2* boudary_map_width)/ boudary_map_width)
        x_list = []
        y_list = []
        # Section 1
        x_list_1 = []
        y_list_1=[]
        for i in range(n_row_points):
            x_list_1.append(random.randint(boudary_map_width, self.map_size_x - boudary_map_width))
            y_list_1.append(random.randint(self.map_size_y - boudary_map_width, self.map_size_y))
        x_list_1.sort()
        x_list += x_list_1
        y_list += y_list_1
        print("Section 1")
        print(x_list)
        print(y_list)
        # Section 2
        x_list.append(random.randint(self.map_size_x - boudary_map_width, self.map_size_x))
        y_list.append(random.randint(self.map_size_y - boudary_map_width, self.map_size_y))
        print("Section 2")
        print(x_list)
        print(y_list)
        # Section 3
        x_list_3 =[]
        y_list_3 =[]
        for i in range(n_row_points):
            x_list_3.append(random.randint(self.map_size_x - boudary_map_width, self.map_size_x))
            y_list_3.append(random.randint(boudary_map_width, self.map_size_y - boudary_map_width))
        y_list_3.sort(reverse=True)
        x_list += x_list_3
        y_list += y_list_3
        print("Section 3")
        print(x_list)
        print(y_list)
        # Section 4
        x_list.append(random.randint(self.map_size_x - boudary_map_width, self.map_size_x))
        y_list.append(random.randint(0, boudary_map_width))
        print("Section 4")
        print(x_list)
        print(y_list)
        # Section 5
        x_list_5 = []
        y_list_5 = []
        for i in range(n_row_points):
            x_list_5.append(random.randint(boudary_map_width, self.map_size_x - boudary_map_width))
            y_list_5.append(random.randint(0, boudary_map_width))
        x_list_5.sort(reverse=True)
        x_list += x_list_5
        y_list += y_list_5
        print("Section 5")
        print(x_list)
        print(y_list)
        # Section 6
        x_list.append(random.randint(0, boudary_map_width))
        y_list.append(random.randint(0, boudary_map_width))
        print("Section 6")
        print(x_list)
        print(y_list)
        # Section 7
        x_list_7 =[]
        y_list_7 =[]
        for i in range(n_row_points):
            x_list_7.append(random.randint(0, boudary_map_width))
            y_list_7.append(random.randint(boudary_map_width, self.map_size_y - boudary_map_width))
        y_list_7.sort()
        x_list += x_list_7
        y_list += y_list_7
        print("Section 7")
        print(x_list)
        print(y_list)
        # Section 8
        x_list.append(random.randint(0, boudary_map_width))
        y_list.append(random.randint(self.map_size_y - boudary_map_width, self.map_size_y))
        print("Section 8")
        print(x_list)
        print(y_list)
        return x_list, y_list



            

