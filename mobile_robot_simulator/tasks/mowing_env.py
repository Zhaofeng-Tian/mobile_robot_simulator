from turtle import done
import gym
from gym.envs.registration import register
import math
from math import sin,cos,atan2,tan,pi,log
import numpy as np
from gym import spaces
import yaml
import sys
sys.path.append('C:\\Users\\61602\\Desktop\\Coding\\python')
from mobile_robot_simulator.world.lawn import Lawn
from mobile_robot_simulator.world.robot import Robot
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
from skimage.draw import disk, polygon


class MowingEnv(gym.Env):
    def __init__(self):
        print("MowingEnv initializing ........")
        with open('C:\\Users\\61602\\Desktop\\Coding\\python\\mobile_robot_simulator\\tasks\\mowing_env.yaml') as file:
            c = yaml.load(file, Loader=yaml.FullLoader)
        self.lawn = Lawn(boundary_flag = 1, boundary_world_width = 1.7, sx = 10,sy = 10,rs = 0.01)
        self.robot = Robot(world = self.lawn, dt = 0.2,v_limits = [[-1,1],[-2.,2.]], 
                            a_limits = [[-1.0,0.5],[-2.,2.]])
        a_high = np.array([self.robot.v_limits[0][1], self.robot.v_limits[1][1]])
        a_low = np.array([self.robot.v_limits[0][0], self.robot.v_limits[1][0]])
        print("a_low: ", a_low)
        self.n_actions = 0
        self.action_space = spaces.Box(a_low.astype(np.float64), a_high.astype(np.float64))
        self.observation_space = spaces.Box(low=0., high=1.0,shape=(100,100), dtype=np.float64)
        self.lawn_area = self.robot.calc_uncut()
        self.uncut = self.lawn_area
        self.cut = 0
        self.step_cntr = 0 ; self.done_cntr = 0
        self.ref_reward = 0
        self.cul_cut = 0
        self.visl = True
        self.p_step = 20
        self.s_list = []
        self.dt = self.robot.dt
        
    
    def set_visl(self,visl):
        self.visl = visl
    def print_env_info(self):
        self.lawn.print_world_info()
        self.robot.print_robot_info()

    def step(self, action):
        self.step_cntr += 1
        # control singal and state update
        # norm_action = self.norm_action(action)
        self.robot.move_base(action)
        self.robot.state_update(visualize = self.visualize)
        self.cut += self.robot.cut_step
        # cut area update
        m = self.robot.mowing_map.copy()
        m = resize(m, self.observation_space.shape[0:2])
        l = self.robot.get_local_map(2*self.observation_space.shape[0])
        l = resize(l, self.observation_space.shape[0:2])
        if self.visl:
            self.visualize(m,l)

        s = np.array(self.robot.state.copy())
        obs_pack = (m,l,s)
        done = self.check_done(s)
        reward = self.calc_reward(done,s)
        if not done:
            self.cur_reward = reward
        info = {}
        return obs_pack, reward,done,info
    
    def visualize(self,m,l):
        #plt.subplot(2,1,1)
        plt.imshow(m ,origin='lower', cmap='gray')
        # plt.subplot(2,1,2)
        # plt.clf()
        # plt.imshow(l,origin='lower', cmap='gray')
        plt.draw()
        plt.pause(0.02)
        # if self. visualize:

        #     plt.clf()
        #     plt.imshow(self.mowing_map, origin='lower', cmap='gray')
        #     plt.draw()
        #     plt.pause(0.02)

    def norm_action(self, action):
        norm_action = action.copy()
        if action[0] >= 0:
            norm_action[0] = action[0]* self.robot.v_limits[0][1]
        else:
            norm_action[0] = -action[0]* self.robot.v_limits[0][0]
        if action[0] >= 0:
            norm_action[1] = action[1]* self.robot.v_limits[1][1]
        else:
            norm_action[1] = -action[1]* self.robot.v_limits[1][0]
        return norm_action
        
    def calc_reward(self,done,state):
        if done:
            reward = 0
            return reward
        else:
            # use percent of cut / lawn area to be the reward
            a = 1/500 # factor to normalize cut area per step
            b = 1
            c = -0.02
            # rate = self.cut/self.lawn_area
            # vs = rate + log(rate)
            # reward = a * self.robot.cut_step + b * vs + c
            reward = a * self.robot.cut_step
            if state[5]:
                reward = -5
                return reward
            else:
                return reward           

    def check_done(self,state):
        self.done_cntr += 1
        done = False
        self.cul_cut += self.robot.cut_step
        if self.step_cntr > 200 and self.step_cntr % 200 == 0 :
            if self.cul_cut <= 20 or self.done_cntr >= 10000:
                self.done_cntr = 0
                done = True
            self.cul_cut = 0
        return done

    def calc_rate(self):
        rate = self.robot.calc_cut()/self.lawn_area
        return rate

    def reset(self):
        self.robot.reset()
        self.step_cntr = 0
        self.cul_cut = 0
        self.cut = 0
        l = self.robot.get_local_map(2*self.observation_space.shape[0])
        l = resize(l, self.observation_space.shape[0:2])
        return self.get_lawn_state(), l, self.get_robot_state()

    def get_obs(self):
        m = self.get_lawn_state
        s = self.get_robot_state
        return (m,s)
        
    def check_crash(self):
        return self.robot.state[-1]
    def reset_world(self):
        self.robot.reset_world()
        return self.robot.world.map
    
    def get_robot_state(self):
        state = np.array(self.robot.state)
        return state

    def get_lawn_state(self):
        m = self.robot.mowing_map.copy()
        m = resize(m, self.observation_space.shape[0:2])
        return m
    def get_world_state(self):
        return self.lawn.map


class MowingEnv_v1(MowingEnv):
    def __init__(self):
        super(MowingEnv_v1,self).__init__()
        print("v1 initalized")
        self.n_actions = 4
        self.action_space = spaces.Discrete(self.n_actions)
        image_path = 'C:\\Users\\61602\\Desktop\\Coding\\python\\mobile_robot_simulator\\maps\\map4.png'
        img = plt.imread(image_path); img = 1 - img
        self.robot.world.map = img.copy()
        self.robot.map_copy = self.robot.world.map.copy()
        self.robot.mowing_map = self.robot.world.map.copy()
        self.robot.init_pose()        
    
    # def set_action(self, a_num):
    #     action = [0.,0.]
    #     if a_num == 0:
    #         action = [0.8,0.]
    #     elif a_num == 1:
    #         action = [0.8, 0.5]
    #     elif a_num == 2:
    #         action = [0.8,-0.5]

    #     elif a_num == 3:
    #         action = [0.,0.5]
    #     elif a_num == 4:
    #         action = [0., -0.5]
    #     elif a_num == 5:
    #         action = [-0.8, 0.]
    #     elif a_num == 6:
    #         action = [-0.8, 0.5]
    #     elif a_num == 7:
    #         action = [-0.8, -0.5]
    #     elif a_num == 8:
    #         action = [0.2,0.]
    #     elif a_num == 9:
    #         action = [-0.2, 0.]
    #     return action
    def set_action(self, a_num):
        action = [0.,0.]
        if a_num == 0:
            action = [0.5,0.]
        elif a_num == 1:
            action = [-0.5, 0.0]
        elif a_num == 2:
            action = [0.0,-0.5]
        elif a_num == 3:
            action = [0.0,0.5]

        return action 

    def general_lawn_state(self, map):
        m = map.copy()
        m = resize(m, self.observation_space.shape[0:2])
        return m
    def general_local_map(self, map_input, state, size):
        x,y = state[0], state[1]
        map_size = size
        map = map_input.copy()
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
        l = map[rrange[0]:rrange[1],crange[0]:crange[1]]
        return resize(l, self.observation_space.shape[0:2])

    def predict(self,map, state, a_num,p_step):
        s_list = []; m = map.copy();s = state.copy()
        #ref_map = self.robot.world.map.copy()
        a = self.set_action(a_num)
        self.robot.move_base(a)
        dt = self.robot.dt
        for i in range(p_step):
            s = self.robot.calc_state(s, dt)
            s_list.append(s)
        # print("s_list: ", s_list)
        self.s_list = s_list
        a_total = []
        for i in range(p_step):
            # print("cal_step: ", self.robot.calc_cut_step(m, s_list[i]))
            a_step = self.robot.calc_cut_step(m, s_list[i])/500 - (5 if s_list[i][5] else 0)
            # print("a_step: ", a_step)

            self.robot.draw_mowing_disk(m)
            a_total.append(a_step)
        # print ("a_total: ", a_total)
        a_sum = sum(a_total)
        return a_sum

    def general_step(self, world, map, m,l,s,a,c):
        map = map.copy()
        world = world.copy()
        s = s.copy()
        m = self.general_lawn_state(map) if m == None else m
        l = self.general_local_map(map, s, size = 2*self.observation_space.shape[0]) if l == None else l
        c = c
        a_real = self.set_action(a)
        self.robot.move_base(a_real)        
        ss = self.robot.calc_state(s, self.dt)
        s_ = s.copy()
        if ss[5]:
            s_[3:] = [0,0,1]
        else:
            s_ = ss
        # vertices calc
        vertices = self.robot.calc_rect_vertices(s_[0],s_[1],s_[2])
        # disk
        row = round(s_[1]/self.robot.world.resolution)
        col = round(s_[0]/self.robot.world.resolution)
        radius = round(self.robot.shape[3]/self.robot.world.resolution)
        d = disk((row,col),radius)
        d0 = np.clip(d[0], 0, 1000)
        d1 = np.clip(d[1], 0, 1000)
        uc = np.sum(world[d0,d1]== 0 )
        c_ = c + uc
        # draw  disk on world map
        world[d0,d1] = 0.5; world_ = world
        map_ = world.copy()
        # draw rectangular on mowing map 
        rr = []
        cc = []        
        for p in vertices:
            rr.append(p[0])
            cc.append(p[1])
        map_[polygon(rr,cc)] = 0.75
        map_[d0,d1] = 0.5

        m_ = self.general_lawn_state(map_)
        l_ = self.general_local_map(map_, s_, size = 2*self.observation_space.shape[0])
        # plt.subplot(2,1,1)
        # plt.imshow(m_ ,origin='lower', cmap='gray')
        # plt.subplot(2,1,2)
        # plt.imshow(l_,origin='lower', cmap='gray')
        # plt.draw()
        # plt.pause(0.02)
        r = -5 if s_[5] else uc/500
        done = 0
        print("********s,******** s_*******")
        print(" old state: ",s,'==> action: ',a , " ==> reward: ", r," ==> new state: ", s_)
        return m,l,s, a, m_,l_,s_,r,done,world_,map_, uc, c_


        
    def visualize(self,m,l):
        plt.clf()
        plt.imshow(self.robot.mowing_map, origin='lower', cmap='gray')
        plt.arrow(round(self.robot.state[0]/self.robot.world.resolution),
                round(self.robot.state[1]/self.robot.world.resolution),
                math.cos(self.robot.state[2])/self.robot.world.resolution,
                math.sin(self.robot.state[2])/self.robot.world.resolution,
                width = 0.1 /self.robot.world.resolution,
                color = 'r')
        plt.draw()
        plt.pause(0.02)


# Test
# env = MowingEnv_v1() # Return a instance of MowingEnv class
# print(env.observation_space.shape[0:2])
# print(env.n_actions)
# print(env.cut)