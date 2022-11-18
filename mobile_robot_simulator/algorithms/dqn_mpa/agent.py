import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from network import DeepQNetwork
from buffer import ReplayBuffer
import random
from collections import deque
import matplotlib.pyplot as plt



class StackAgent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, env, state_shape,
                 mem_size=100000,local_mem_size = 128, batch_size=64, eps_min=0.01, eps_dec=5e-7, stack_steps = 4,
                 replace=1000, policy='None',chkpt_dir='C:\\Users\\61602\\Desktop\\Coding\\python\\mobile_robot_simulator\\algorithms\\ddqn\\plots\\models\\'):
        self.gamma = gamma
        self.epsilon = epsilon;         self.local_mem_size = local_mem_size
        self.lr = lr
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.input_dims = input_dims
        self.tensor_shape = (-1,input_dims[0],input_dims[1], stack_steps)
        self.rs_tensor_shape = (-1, stack_steps*6)
        self.batch_size = batch_size
        self.policy = policy
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.fname = self.chkpt_dir
        self.env =env
        self.mpc_range = range(0,1);    self.collision_range = range(0,1)
        self.random_range = range(-1,0); self. current_action = 0
        self.explore_str = 0;           self.multi_collisions = False
        self.episode_step_str = 0;      self.stack_steps = stack_steps
        self.check_cut = False;         self.cut_area = 0

        self.memory = ReplayBuffer(mem_size, self.tensor_shape[1:], self.rs_tensor_shape[1:],n_actions)
        self.local_memory = ReplayBuffer(self.local_mem_size, self.tensor_shape[1:], self.rs_tensor_shape[1:],n_actions)

        self.q_eval = DeepQNetwork(input_dims, state_shape, n_actions)
        self.q_eval.compile(optimizer=Adam(learning_rate=lr))
        self.q_next = DeepQNetwork(input_dims, state_shape, n_actions)
        self.q_next.compile(optimizer=Adam(learning_rate=lr))

    def save_models(self):
        self.q_eval.save(self.fname+'q_eval')
        self.q_next.save(self.fname+'q_next')
        print('... models saved successfully ...')

    def load_models(self):
        self.q_eval = keras.models.load_model(self.fname+'q_eval')
        self.q_next = keras.models.load_model(self.fname+'q_next')
        print('... models loaded successfully ...')

    def store_transition(self,mem_type,obs,lmap,state, action, reward, obs_, lmap_,state_, done):
        
        # obs = tf.convert_to_tensor(obs, dtype=tf.float32); obs = tf.reshape(obs, self.tensor_shape[1:]) 
        # lmap = tf.convert_to_tensor(lmap, dtype=tf.float32); lmap = tf.reshape(lmap, self.tensor_shape[1:])
        # state = tf.convert_to_tensor(state, dtype=tf.float32);state = tf.reshape(state, self.rs_tensor_shape[1:])
        # obs_ = tf.convert_to_tensor(obs_, dtype=tf.float32); obs_ = tf.reshape(obs_, self.tensor_shape[1:]) 
        # lmap_ = tf.convert_to_tensor(lmap_, dtype=tf.float32); lmap_ = tf.reshape(lmap_, self.tensor_shape[1:])
        # state_ = tf.convert_to_tensor(state_, dtype=tf.float32);state_ = tf.reshape(state_, self.rs_tensor_shape[1:])

        obs = np.array(obs).astype('float32'); obs = obs.reshape(self.tensor_shape[1:])      
        lmap = np.array(lmap).astype('float32'); lmap = lmap.reshape(self.tensor_shape[1:])  
        state = np.array(state).astype('float32');state = state.reshape(self.rs_tensor_shape[1:]) 
        obs_ = np.array(obs_).astype('float32'); obs_ = obs_.reshape(self.tensor_shape[1:])  
        lmap_ = np.array(lmap_).astype('float32'); lmap_ = lmap_.reshape(self.tensor_shape[1:])
        state_ = np.array(state_).astype('float32');state_ = state_.reshape(self.rs_tensor_shape[1:])
        print(" storing obs shape : ", obs.shape)
        if mem_type == 'global':
            self.memory.store_transition(obs, lmap, state, action, reward, obs_, lmap_, state_, done)
        elif mem_type == 'local':
            self.local_memory.store_transition(obs, lmap, state, action, reward, obs_, lmap_, state_, done)
        else:
            print(" Memory Type Wrong!")

    # def sample_memory(self):
    #     station, action, reward, new_state, done = \
    #                               self.memory.sample_buffer(self.batch_size)
    #     states = tf.convert_to_tensor(state)
    #     rewards = tf.convert_to_tensor(reward)
    #     dones = tf.convert_to_tensor(done)
    #     actions = tf.convert_to_tensor(action, dtype=tf.int32)
    #     states_ = tf.convert_to_tensor(new_state)
    #     return states, actions, rewards, states_, dones

    def choose_action(self, obs, lmap, state, policy, stack_ctr):
        self.cut_area += self.env.robot.cut_step
        self.check_cut = True if stack_ctr % 20 == 0 else False
        if policy=="ddqn":
            obs = tf.convert_to_tensor(obs, dtype=tf.float32); obs = tf.reshape(obs, self.tensor_shape) 
            lmap = tf.convert_to_tensor(lmap, dtype=tf.float32); lmap = tf.reshape(lmap, self.tensor_shape)
            state = tf.convert_to_tensor(state, dtype=tf.float32);state = tf.reshape(state, self.rs_tensor_shape)
            # print('in choose_action, obs:', str(obs))
            # print('in choose_action, s: ', str(s))
            actions = self.q_eval((obs,lmap,state))
            # print('actions after q_eval: ', actions)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
            # print('action after argmax: ', action)
            return action

        elif policy=="random":
            v_reward = np.zeros((self.n_actions,self.n_actions))
            v_q = np.zeros((self.n_actions,self.n_actions))
            v_r1r2 = [0.,0.]
            for i in range(self.n_actions):
                print(" ************ In map, checking action: ", i)
                world = self.env.robot.world.map.copy()
                map = self.env.robot.mowing_map.copy()
                s = self.env.robot.state.copy()
                c = self.env.cut
                mm = obs.copy(); ll=lmap.copy(); ss=state.copy()
                mm_ = deque(maxlen=self.stack_steps)
                ll_ = deque(maxlen=self.stack_steps)
                ss_ = deque(maxlen=self.stack_steps)
                rr  = deque(maxlen = self.stack_steps)
                for j in range(self.stack_steps):
                    m,l,s,a,m_,l_,s_,r,done,world_,map_, uc, c_ = self.env.general_step(world, map, None, None,s, i, c)
                    # plt.subplot(2,1,1)
                    # plt.imshow(m_ ,origin='lower', cmap='gray')
                    # plt.subplot(2,1,2)
                    # plt.imshow(l_ ,origin='lower', cmap='gray')
                    # plt.draw()
                    # plt.pause(3)

                    mm_.append(m_);ll_.append(l_);ss_.append(s_); rr.append(r)
                    world = world_.copy(); map = map_.copy(); s = s_.copy(); c = c_
                r1 = sum(rr); v_r1r2[0] =r1
                self.store_transition('local',mm,ll,ss,a,r,mm_,ll_,ss_,done)
                self.learn('local')

                for ii in range(self.n_actions):
                    world2 = world_.copy()
                    map2 = map_.copy()
                    s2 = s_.copy()
                    c2 = c_
                    mm2 = mm_.copy(); ll2 = ll_.copy(); ss2 = ss_.copy()
                    mm2_ = deque(maxlen=self.stack_steps)
                    ll2_ = deque(maxlen=self.stack_steps)
                    ss2_ = deque(maxlen=self.stack_steps)
                    rr2  = deque(maxlen = self.stack_steps) 
                    for jj in range(self.stack_steps):
                        
                        m2,l2,s2,a2,m2_,l2_,s2_,r2,done2,world2_,map2_, uc2, c2_ = self.env.general_step(world2, map2, None, None,s2, ii, c2)
                        # plt.subplot(2,1,1)
                        # plt.imshow(m2_ ,origin='lower', cmap='gray')
                        # plt.subplot(2,1,2)
                        # plt.imshow(l2_ ,origin='lower', cmap='gray')
                        # plt.draw()
                        # plt.pause(3)

                        mm2_.append(m2_);ll2_.append(l2_);ss2_.append(s2_); rr2.append(r2) 
                        world2 = world2_.copy(); map2 = map2_.copy(); s2 = s2_.copy(); c2 = c2_ 
                    r2 = sum(rr2); v_r1r2[1] = r2 
                    self.store_transition('local',mm2,ll2,ss2,a2,r2,mm2_,ll2_,ss2_,done2)
                    self.learn('local')

            if stack_ctr in self.random_range:
                action = self.current_action
                return action
            else:
                action = np.random.choice(range(0,self.n_actions))
                if self.check_cut and self.cut_area < 20:
                    self.random_range = range(stack_ctr,stack_ctr+5)
                else:
                    if stack_ctr <= 1000:
                        self.random_range = range(stack_ctr, stack_ctr+1)
                    elif stack_ctr - self.episode_step_str >= 2000:
                        self.random_range = range(stack_ctr, stack_ctr+3)
                    else:
                        self.random_range = range(stack_ctr, stack_ctr+2)
                self.current_action = action
                return action

        elif policy == "collision":
            f = random.uniform(0,1)
            action = 0 if f > 0.5 else 1
            return action

        elif policy == "mpa":
            actions = []
            for i in range(self.n_actions):
                print(" ************ In map, checking action: ", i)
                world = self.env.robot.world.map.copy()
                map = self.env.robot.mowing_map.copy()
                s = self.env.robot.state.copy()
                c = self.env.cut
                mm = obs.copy(); ll=lmap.copy(); ss=state.copy()
                mm_ = deque(maxlen=self.stack_steps)
                ll_ = deque(maxlen=self.stack_steps)
                ss_ = deque(maxlen=self.stack_steps)
                rr  = deque(maxlen = self.stack_steps)
                for j in range(self.stack_steps):
                    m,l,s,a,m_,l_,s_,r,done,world_,map_, uc, c_ = self.env.general_step(world, map, None, None,s, i, c)
                    # plt.subplot(2,1,1)
                    # plt.imshow(m_ ,origin='lower', cmap='gray')
                    # plt.subplot(2,1,2)
                    # plt.imshow(l_ ,origin='lower', cmap='gray')
                    # plt.draw()
                    # plt.pause(5)

                    mm_.append(m_);ll_.append(l_);ss_.append(s_); rr.append(r)
                    world = world_.copy(); map = map_.copy(); s = s_.copy(); c = c_
                r = sum(rr); actions.append(r)
                # plt.subplot(8,1,1)
                # plt.imshow(mm[0] ,origin='lower', cmap='gray')
                # plt.subplot(8,2,1)
                # plt.imshow(mm[1] ,origin='lower', cmap='gray')
                # plt.subplot(8,3,1)
                # plt.imshow(mm[2] ,origin='lower', cmap='gray')
                # plt.subplot(8,4,1)
                # plt.imshow(mm[3] ,origin='lower', cmap='gray')
                # plt.subplot(8,1,2)
                # plt.imshow(mm_[0] ,origin='lower', cmap='gray')
                # plt.subplot(8,2,2)
                # plt.imshow(mm_[1] ,origin='lower', cmap='gray')
                # plt.subplot(8,3,2)
                # plt.imshow(mm_[2] ,origin='lower', cmap='gray')
                # plt.subplot(8,4,2)
                # plt.imshow(mm_[3] ,origin='lower', cmap='gray')
                # plt.draw()
                # plt.pause(5)
                self.store_transition(mm,ll,ss,a,r,mm_,ll_,ss_,done)
            return np.argmax(actions)

        elif policy == "mpa2":
            v_reward = np.zeros((self.n_actions,self.n_actions))
            v_q = np.zeros((self.n_actions,self.n_actions))
            v_r1r2 = [0.,0.]
            for i in range(self.n_actions):
                print(" ************ In map, checking action: ", i)
                world = self.env.robot.world.map.copy()
                map = self.env.robot.mowing_map.copy()
                s = self.env.robot.state.copy()
                c = self.env.cut
                mm = obs.copy(); ll=lmap.copy(); ss=state.copy()
                mm_ = deque(maxlen=self.stack_steps)
                ll_ = deque(maxlen=self.stack_steps)
                ss_ = deque(maxlen=self.stack_steps)
                rr  = deque(maxlen = self.stack_steps)
                for j in range(self.stack_steps):
                    m,l,s,a,m_,l_,s_,r,done,world_,map_, uc, c_ = self.env.general_step(world, map, None, None,s, i, c)
                    # plt.subplot(2,1,1)
                    # plt.imshow(m_ ,origin='lower', cmap='gray')
                    # plt.subplot(2,1,2)
                    # plt.imshow(l_ ,origin='lower', cmap='gray')
                    # plt.draw()
                    # plt.pause(3)

                    mm_.append(m_);ll_.append(l_);ss_.append(s_); rr.append(r)
                    world = world_.copy(); map = map_.copy(); s = s_.copy(); c = c_
                r1 = sum(rr); v_r1r2[0] =r1
                self.store_transition('local',mm,ll,ss,a,r,mm_,ll_,ss_,done)
                self.learn('local')

                for ii in range(self.n_actions):
                    world2 = world_.copy()
                    map2 = map_.copy()
                    s2 = s_.copy()
                    c2 = c_
                    mm2 = mm_.copy(); ll2 = ll_.copy(); ss2 = ss_.copy()
                    mm2_ = deque(maxlen=self.stack_steps)
                    ll2_ = deque(maxlen=self.stack_steps)
                    ss2_ = deque(maxlen=self.stack_steps)
                    rr2  = deque(maxlen = self.stack_steps) 
                    for jj in range(self.stack_steps):
                        
                        m2,l2,s2,a2,m2_,l2_,s2_,r2,done2,world2_,map2_, uc2, c2_ = self.env.general_step(world2, map2, None, None,s2, ii, c2)
                        # plt.subplot(2,1,1)
                        # plt.imshow(m2_ ,origin='lower', cmap='gray')
                        # plt.subplot(2,1,2)
                        # plt.imshow(l2_ ,origin='lower', cmap='gray')
                        # plt.draw()
                        # plt.pause(3)

                        mm2_.append(m2_);ll2_.append(l2_);ss2_.append(s2_); rr2.append(r2) 
                        world2 = world2_.copy(); map2 = map2_.copy(); s2 = s2_.copy(); c2 = c2_ 
                    r2 = sum(rr2); v_r1r2[1] = r2 
                    self.store_transition('local',mm2,ll2,ss2,a2,r2,mm2_,ll2_,ss2_,done2)
                    self.learn('local')

                    print("ss2_ : ", ss2_)
                    f_obs = tf.convert_to_tensor(mm2_, dtype=tf.float32); f_obs = tf.reshape(f_obs, self.tensor_shape) 
                    f_lmap = tf.convert_to_tensor(ll2_, dtype=tf.float32); f_lmap = tf.reshape(f_lmap, self.tensor_shape)
                    f_state = tf.convert_to_tensor(ss2_, dtype=tf.float32);f_state = tf.reshape(f_state, self.rs_tensor_shape)
                    # print('in choose_action, obs:', str(obs))
                    # print('in choose_action, s: ', str(s))
                    qvalues = self.q_eval((f_obs,f_lmap,f_state)) ;print("action 1: ",i,"  action 2: ", ii, "==> Q values: ", qvalues)
                    # svalue = sum(qvalues.numpy())/self.n_actions
                    svalue = tf.reduce_sum(qvalues).numpy()/self.n_actions
                    print(" svalue: " ,svalue)
                    r_sum = sum(v_r1r2)
                    v_q[i][ii] = svalue;    print("action 1: ",i,"  action 2: ", ii, "==> state value: ", svalue)
                    v_reward[i][ii] = r_sum; print("action 1: ",i,"  action 2: ", ii, "==> r sum: ", r_sum)
                    v_two = v_reward+v_q

            for r in range(self.n_actions):
                for c in range(self.n_actions):
                    print(r,c,'==>',v_two[r][c], end=' ')

            action = np.argmax(v_reward+v_q); print("action index: ", action)
            action = int(action/self.n_actions); print("action output: ", action)



            return action




        elif policy=="model_predictive":
            map = self.env.robot.world.map.copy()
            s = state.copy()
            p_step =self.env.p_step
            actions = []
            for i in range(self.n_actions):
                actions.append(self.env.predict(map,s,i,p_step))
            print("action advantages: " ,actions)
            action = actions.index(max(actions))
            return action
        if self.check_cut:
            self.cut_area = 0    

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def learn(self, mem_type):

        # print(" *************** Agent is Learning ! ! *********************")
        # self.replace_target_network()
        if mem_type == 'global':
            print(" Gloabal Memory Ctr: ",self.memory.mem_cntr)
            if self.memory.mem_cntr < self.batch_size:
                return
            obs, lmap, state, action, reward,obs_, lmap_, state_, done = \
                self.memory.sample_buffer(self.batch_size)
            print("7777777777777 Global Learning 777777777777777777")
        elif mem_type == 'local':
            print(" Local Memory Ctr: ",self.local_memory.mem_cntr)
            if self.local_memory.mem_cntr < self.batch_size:
                return
            print(" 888888888888 Local Learning 8888888888888")
            obs, lmap, state, action, reward,obs_, lmap_, state_, done = \
                self.local_memory.sample_buffer(self.batch_size)
        else:
            print(" Memory Type Error When Sampling Buffer!!")
        # print(len(obs))
        # print("obs: ", obs)
        # print(obs.shape)
        # obs to tensor
        obs = tf.convert_to_tensor(obs,dtype=tf.float32)
        obs = tf.reshape(obs,self.tensor_shape)
        # print("obs shape: ", obs.shape)
        # obs_
        obs_ = tf.convert_to_tensor(obs_, dtype=tf.float32)
        obs_ = tf.reshape(obs_,self.tensor_shape)
        # lmap
        lmap = tf.convert_to_tensor(lmap, dtype=tf.float32)
        lmap = tf.reshape(lmap,self.tensor_shape)
        # lmap_
        lmap_ = tf.convert_to_tensor(lmap_, dtype=tf.float32)
        lmap_ = tf.reshape(lmap_,self.tensor_shape)
        # s
        s = tf.convert_to_tensor(state, dtype=tf.float32)
        s = tf.reshape(s, self.rs_tensor_shape)
        # s_
        s_ = tf.convert_to_tensor(state_, dtype=tf.float32)
        # print("s_ shape ", s_.shape)
        # print("s_ before reshape: ", s_)
        s_ = tf.reshape(s_, self.rs_tensor_shape)
        # print("s_ :", s_)
        
        # reward
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        # action
        actions = tf.convert_to_tensor(action, dtype=tf.int32)

        # states, actions, rewards, states_, dones = self.sample_memory()

        indices = tf.range(self.batch_size, dtype=tf.int32)
        action_indices = tf.stack([indices, actions], axis=1)

        with tf.GradientTape() as tape:
            q_pred = tf.gather_nd(self.q_eval((obs, lmap,s)), indices=action_indices)
            q_next = self.q_next((obs_, lmap_, s_))
            q_eval = self.q_eval((obs_, lmap_, s_))

            max_actions = tf.math.argmax(q_eval, axis=1, output_type=tf.int32)
            max_action_idx = tf.stack([indices, max_actions], axis=1)

            q_target = reward + \
                self.gamma*tf.gather_nd(q_next, indices=max_action_idx) *\
                (1 - done)

            loss = keras.losses.MSE(q_pred, q_target)

        params = self.q_eval.trainable_variables
        grads = tape.gradient(loss, params)

        self.q_eval.optimizer.apply_gradients(zip(grads, params))

        self.learn_step_counter += 1

        self.decrement_epsilon()
        self.replace_target_network()