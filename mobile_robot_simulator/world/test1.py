from car_robot import CarRobot
from mobile_robot_simulator.world.world import World
from mobile_robot_simulator.world.param import CarParam

import numpy as np
import matplotlib.pyplot as plt
import time

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

#         # coord = [y, x] if is_steep else (x, y)
#         # points.append(coord)
#         # error -= abs(dy)
#         # if error < 0:
#         #     y += y_step
#         #     error += dx
#         # if map[x][y] >= 0.5 and (map[x][y]>= thresh+0.005 or map[x][y]<=thresh-0.005):
#         #     print(" breaking la!!!",x,y, map[x][y])
#         #     break;
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
param = CarParam()
res = param.world_resolution 
world = World(size_x = 20, size_y = 20, resolution = res )
map = world.map.copy()
robot = CarRobot(np.array([5.,5.,2.15,0.,0.]),param,0)
robot2 = CarRobot(np.array([5.,9.,1.5708,0.,0.]),param,1)
robot3 = CarRobot(np.array([7.,5.,1.5708,0.,0.]),param,2)
print(robot.disc_centers)
robot.fill_rect_body(robot.vertices,map, robot.thresh)
robot.fill_rect_body(robot2.vertices,map, robot2.thresh)
robot.fill_rect_body(robot3.vertices,map, robot3.thresh)

# print(CarRobot.build_line(CarRobot.p2i(robot.disc_centers[0],res),CarRobot.p2i(robot.disc_centers[1],res)))
# print(CarRobot.p2i(robot.disc_centers[0],res))
# print(CarRobot.p2i(robot.disc_centers[1],res))
# line = CarRobot.build_line(CarRobot.p2i(robot.disc_centers[0],res),CarRobot.p2i(robot.disc_centers[1],res))
# ray = build_a_ray((280,250),(910,250),map,robot.thresh)
# print(map[280][250])

# ends = robot.build_ray_ends()
# print("ends: ", ends)
# rays = []
# lidar_cell = robot.p2i(robot.disc_centers[2],robot.resolution)
# print("**********", lidar_cell)
# for (r,c) in ends:
#     rays.append(build_a_ray(lidar_cell,(r,c),map,robot.thresh))
# rays,obs = robot.get_scans(map)
# print(obs)
# print("how many rays? ", len(rays))

for step in range(0,5):
    start = time.time()
    map= world.map.copy()

    robot.move_base([10,10]); robot2.move_base([0,0]); robot3.move_base([0,0])
    # print(robot.control_signal)
    robot.state_update(map)
    robot2.state_update(map)
    robot3.state_update(map)
    # print(robot.state)
    # print(robot.vertices)
    robot.fill_rect_body(robot.vertices,map, robot.thresh)
    robot2.fill_rect_body(robot2.vertices,map, robot2.thresh)
    robot3.fill_rect_body(robot3.vertices,map, robot3.thresh)

    robot.get_scans(map)
    robot2.get_scans(map)
    robot3.get_scans(map)

    rays = robot.rays
    end = time.time()
    print("one step time: ", end-start)
    # print(robot.obs)

    # ray = robot.build_a_ray( robot.p2i(robot.disc_centers[2],res), robot.p2i(robot.disc_centers[1],res),map, robot.thresh)
    # print("ray: ", ray)
    # plt.clf()
    # plt.imshow(map, origin='lower', cmap='gray')
    # # plt.draw()
    # for ray in rays:
    #     line_x = [p[1] for p in ray]
    #     line_y = [p[0] for p in ray]
    #     # print(ray[-1])
    #     plt.plot(line_x,line_y)
    # plt.show()
    # plt.pause(0.01)