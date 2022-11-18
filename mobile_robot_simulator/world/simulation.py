import math
import sys
sys.path.append('C:\\Users\\61602\\Desktop\\Coding\\python')
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import random

from mobile_robot_simulator.world.lawn import Lawn
from mobile_robot_simulator.world.robot import Robot
from skimage.io import imsave, imread
from skimage.transform import rescale, resize, downscale_local_mean

""" 
About map layers: 
static obstacles with lawn boundaries are ploted on "lawn.map"
iteration begins:
    copy "lawn.map" to "mowing_map"
    
    calculate robot pose and whether will collide with "lawn.map"
    whether collide or not, plot it on "mowing_map"
"""

lawn = Lawn()
map_copy = lawn.map
lawn.print_world_info()
print("Map size: " , lawn.map_size_x, lawn.map_size_y)
# plt.clf()
# plt.imshow(lawn.map, origin='lower')
# plt.show()
robot = Robot(world = lawn,dt = 0.2)
pose_copy = robot.state[:3]
robot.print_robot_info()
p1,p2,p3,p4 = robot.calc_rect_vertices(robot.state[0],robot.state[1],robot.state[2])
print("p1: "+ str(p1))
pts=[p1,p2,p3,p4]
print("pts: " + str(pts))
img = 1-lawn.map
imsave("map.png", img)
mowing_map = lawn.map.copy()
robot.fill_rect_body(pts, mowing_map)

def get_local_map(map,coord,map_size=200):
    x,y = coord
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

    print(rrange,crange)
    return rrange, crange
   
# mahotas.polygon.fill_polygon(pts, lawn.map) 

robot.print_robot_info()
robot.move_base([10,0.0])
# Plot setting
free = np.array([1, 1, 1, 1])
occ = np.array([0, 0, 0, 1])
unknown = np.array([0.75, 0.9, 0.95, 1])
world_cmp = ListedColormap(np.vstack((free, occ)))
print(world_cmp)
robot_cmp = ListedColormap(np.vstack((free, occ, unknown)))
print(robot_cmp)
norm = plt.Normalize(0,1)
i = 0
#plt.figure()
while True:
    print("Please input 1 or 2 to continue or reset:")
    k = input()
    if k =="w":
        i += 1
        print(" Interation: ", i)
        """ I. Publish new control signal """
        robot.move_base([1.0,0.0])

        """ II. Update the state """
        robot.state_update()
        
        print ("Uncut area: " ,robot.calc_uncut())
        print ("Cut area: ", robot.calc_cut())

        """ III. Draw robot and mowing area """
        # img = resize(image, (image.shape[0] // 4, image.shape[1] // 4),anti_aliasing=True)
        #img = resize(robot.mowing_map,(100,100),anti_aliasing=True)

        # plt.imshow(img, origin='lower', cmap='gray')
    elif k == "a":
        i += 1
        robot.move_base([1.0,1.0])
        robot.state_update()
        #lmap = get_local_map(robot.mowing_map.copy(), (robot.state[0],robot.state[1]),map_size=200)
        print (robot.state[0:2])
        # r,c = get_local_map(robot.mowing_map.copy(), (robot.state[0],robot.state[1]),map_size=200)
        #print(r,c)
        m = robot.mowing_map.copy()
        lmap = robot.get_local_map(200)
        m = resize(m, (128,128))
        lmap = resize(lmap, (128,128))
        print("local map shape: ", lmap.shape)

        plt.subplot(2,1,1)
        plt.imshow(m ,origin='lower', cmap='gray')
        #plt.draw()
        plt.subplot(2,1,2)
        # plt.clf()
        plt.imshow(lmap,origin='lower', cmap='gray')
        plt.draw()
        plt.pause(0.01)


    
    elif k == "d":
        i += 1
        robot.move_base([1.0,-1.0])
        robot.state_update()
    elif k == "s":
        i += 1
        robot.move_base([-1.0,0.0])
        robot.state_update()

    

    
    elif k == "2":
        robot.reset()
        plt.clf()
        plt.imshow(robot.mowing_map, origin='lower', cmap='gray')
        plt.draw()
        plt.pause(0.02)

    else:
        break





robot.body_collision_check(1,1,0)
robot.body_collision_check(0.2,0.2,0)