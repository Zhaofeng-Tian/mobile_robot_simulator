from turtle import done
import gym
from gym.envs.registration import register
from math import sin,cos,atan2,tan,pi,log
import numpy as np
from gym import spaces
import yaml
import sys
sys.path.append('C:\\Users\\61602\\Desktop\\Coding\\python')
from mobile_robot_simulator.world.lawn import Lawn
from mobile_robot_simulator.world.robot import Robot
from skimage.transform import rescale, resize, downscale_local_mean


class MowingEnv(gym.Env):
    def __init__(self):
        print("MowingEnv initializing ........")
        with open('C:\\Users\\61602\\Desktop\\Coding\\python\\mobile_robot_simulator\\tasks\\mowing_env.yaml') as file:
            c = yaml.load(file, Loader=yaml.FullLoader)
        self.lawn = Lawn(boundary_flag = c['lawn']['bf'], 
                    boundary_world_width = c['lawn']['bww'],
                    sx = c['lawn']['sx'],
                    sy = c['lawn']['sy'],
                    rs = c['lawn']['rs'])
        self.robot = Robot(world = self.lawn,
                        dt = 0.2,
                        v_limits = c['robot']['v_limits'],
                        a_limits = c['robot']['a_limits'])
        a_high = np.array([self.robot.v_limits[0][1], self.robot.v_limits[1][1]])
        a_low = np.array([self.robot.v_limits[0][0], self.robot.v_limits[1][0]])
        print("a_low: ", a_low)
        self.action_space = spaces.Box(a_low.astype(np.float64), a_high.astype(np.float64))
        self.observation_space = spaces.Box(low=0., high=1.0,shape=(200,200), dtype=np.float64)
        self.lawn_area = self.robot.calc_uncut()
        self.uncut = self.lawn_area
        self.cut = 0
        self.step_cntr = 0 ; self.done_cntr = 0
        self.ref_reward = 0
        self.cul_cut = 0
        self.visualize = True
    

    def print_env_info(self):
        self.lawn.print_world_info()
        self.robot.print_robot_info()

    def step(self, action):
        self.step_cntr += 1
        # control singal and state update
        norm_action = self.norm_action(action)
        self.robot.move_base(norm_action)
        self.robot.state_update(visualize = self.visualize)
        self.cut += self.robot.cut_step
        # cut area update
        m = self.robot.mowing_map.copy()
        m = resize(m, self.observation_space.shape[0:2])
        s = np.array(self.robot.state.copy())
        obs_pack = (m,s)
        done = self.check_done(s)
        reward = self.calc_reward(done,s)
        if not done:
            self.cur_reward = reward
        info = {}
        return obs_pack, reward,done,info
    
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
            rate = self.cut/self.lawn_area
            vs = rate + log(rate)
            reward = a * self.robot.cut_step + b * vs + c
            if state[5]:
                reward += -1
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
        return self.get_lawn_state(), self.get_robot_state()

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



    