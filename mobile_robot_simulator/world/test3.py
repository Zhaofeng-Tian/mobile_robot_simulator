import imageio
import numpy as np
import matplotlib.pyplot as plt
from car_robot import CarRobot
from mobile_robot_simulator.world.world import World
from mobile_robot_simulator.world.param import CarParam
from reeds_shepp_path_planning import reeds_shepp_path_planning
import time

img = imageio.imread('racetrack.png')
map = np.array(img)
map = 1.0-map.astype(float)/255

param = CarParam()
world = World(size_x = 100, size_y = 100, resolution = 0.01)
robot = CarRobot(np.array([27,4.8,1.5708,0.,0.]),param,0)
robot.fill_rect_body(robot.vertices,map, robot.thresh)
robot.get_scans(map)
start_x = 27  # [m]
start_y =4.8  # [m]
start_yaw = 1.5708  # [rad]

end_x = 27  # [m]
end_y = 40  # [m]
end_yaw = 2.3  # [rad]

curvature = 0.7
step_size = 0.02
st = time.time()
xs, ys, yaws, modes, lengths = reeds_shepp_path_planning(start_x, start_y,
                                                            start_yaw, end_x,
                                                            end_y, end_yaw,
                                                            curvature,
                                                            step_size)
xs = np.array(xs)*100; ys =  np.array(ys)*100
et = time.time()
print(xs)
plt.clf()
plt.imshow(map, origin='lower', cmap='gray')
plt.plot(xs, ys, label="final course " + str(modes))
# for ray in robot.rays:
#     line_x = [p[1] for p in ray]
#     line_y = [p[0] for p in ray]
#     plt.plot(line_x,line_y)
# plt.draw()
# plt.pause(0.01)
# plt.imshow(map, origin='lower', cmap='gray')
plt.show()
print("time for path: ", et-st)