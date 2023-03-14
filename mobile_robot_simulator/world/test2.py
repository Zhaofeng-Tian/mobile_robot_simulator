from car_robot import CarRobot
from mobile_robot_simulator.world.world import World
from mobile_robot_simulator.world.param import CarParam
from mobile_robot_simulator.world.utils import plot_car
import numpy as np
import matplotlib.pyplot as plt
import time

param = CarParam()
world = World(size_x = 100, size_y = 100, resolution = 0.02)

robots = []
for i in range(100):
    robot = CarRobot(np.array([5+0.8*i,5+0.8*i,0.,0.,0.]),param,i)
    robots.append(robot)

for i in range(int(60/robot.dt)):

    map = world.map.copy()
    start = time.time()
    for robot in robots:
        robot.move_base([10,10])
        # print(robot.control_signal)
        robot.state_update(map)
        # print(robot.state)
        # print(robot.vertices)
        robot.fill_rect_body(robot.vertices,map,robot.thresh)
    for j in range(100):
        robots[j].get_scans(map)
    end = time.time()
    print("one update total_time: ", end-start)

    plt.clf()
    plt.imshow(map, origin='lower', cmap='gray')
    for ray in robots[2].rays:
        line_x = [p[1] for p in ray]
        line_y = [p[0] for p in ray]
        plt.plot(line_x,line_y)
    plt.draw()
    plt.pause(0.01)

print(len(robot.sequence))