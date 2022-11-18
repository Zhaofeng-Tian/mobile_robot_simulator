import gym
import numpy as np
from agent import StackAgent
from utils import plot_learning_curve, manage_memory, save_loss, save_reward, save_score
import random
import sys
sys.path.append('C:\\Users\\61602\\Desktop\\Coding\\python')
from mobile_robot_simulator.tasks.register import Make_Env
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Global setting

    manage_memory()
    pack_file = 'C:\\Users\\61602\\Desktop\\Coding\\python\\mobile_robot_simulator\\algorithms\\dqn_mpa\\plots\\'
    visualize = True
    # env and agent initialize
    env = Make_Env('Mowing-v1')  # v1 is a discrete env
    env.set_visl(visualize)
    stack_steps = 4
    print("****************init agent*******************")
    agent = StackAgent(gamma=0.995,epsilon=0.9, lr = 1e-4,
                input_dims=[env.observation_space.shape[0],env.observation_space.shape[1]], env=env,
                n_actions= env.n_actions,
                state_shape=6,mem_size=30000, local_mem_size = 128, batch_size=64,
                eps_min =0.01, eps_dec=5e-7, stack_steps = stack_steps,
                replace = 1000, chkpt_dir= pack_file + 'models\\')
    print('************* agent successfully init ************')
    print('Agent lOCAL Mem Size: ', agent.local_mem_size)
              
    # train mode initialize
    load_checkpoint = False
    load_train = False

    # train mode logic setting
    if load_checkpoint:
        agent.load_models()
        evaluate = True
    elif load_train:
        agent.load_models()
        evaluate = False
    else:
        evaluate = False 
    
    # game variables initialize
    best_score = -100000
    score_list=[];step_list=[];episode_list=[]
    actor_loss=[];critic_loss=[]
    reward_list=[];interval_list=[];average_reward=[];coverage_list=[]
    stack_list = []
    visualize = True
    step_ctr = 0; interval_ctr = 0; stack_ctr = 0
    n_games = 100; n_steps = 10000; 

    # *******************  Trainning  *************************
    for i in range(n_games):
        episode_list.append(i+1)
        # parameters initialized
        stack_ctr = 0;done = False; score = 0;ref_step = 0
        random_steps = range(ref_step,ref_step+5)
        policy = "random"; action = 0
        # 1 ==> states initialized m, l, s
        m, l, s = env.reset() ; # prestate = s.copy()
        # 2 ==> deques initialized mm, ll, ss
        mm = deque(maxlen = stack_steps); ll = deque(maxlen = stack_steps)
        ss = deque(maxlen = stack_steps); rr = deque(maxlen = stack_steps)
        collision_que = deque(maxlen = stack_steps)
        for step in range(stack_steps):
            mm.append(m); ll.append(l); ss.append(s); rr.append(0)
            collision_que.append(0)
        # 3 ==> copy mm, ll , ss for next frame
        mm_ = mm.copy(); ll_= ll.copy(); ss_ = ss.copy()

        
        # ************************  Episode/ Game *************************
        while not done:
            step_ctr += 1; interval_check = step_ctr % 30
            step_list.append(step_ctr)
            collision_que.append(s[5])
            

            # ************************* Normal Step Start  ******************************
            action_real = env.set_action(action)
            obs_pack, reward, done, info = env.step(action_real)
            m, l, s = obs_pack
            # update some info
            collision_que.append(s[5])
            reward_list.append(reward)
            mm_.append(m); ll_.append(l); ss_.append(s); rr.append(reward) #m = m_
            r = sum(rr)

            if interval_check == 0:
                interval_list.append(interval_ctr+1)
                average_reward.append(np.mean(reward_list[-30:]))
                coverage_list.append(env.calc_rate())
                interval_ctr += 1 

            # stack ctr update and policy update
            if step_ctr % 4 ==0:
                # ********************* Stack Step Start ***************************
                stack_ctr += 1 
                r = sum(rr) ; score += r 
                print("Episode[{}/{}] ==> Step[{}/{}] ==> Policy {} ==> Action {} ==> Reward {:.4f}"
                    .format(i+1, n_games, step_ctr, n_steps, policy, str(action), r)) 

                agent.store_transition('global',mm, ll, ss , action, r,
                                       mm_, ll_, ss_, done) 
                # visualize
                # plt.subplot(4,1,1)
                # plt.imshow(mm[0] ,origin='lower', cmap='gray')
                # plt.subplot(4,1,2)
                # plt.imshow(mm[1] ,origin='lower', cmap='gray')
                # plt.subplot(4,1,3)
                # plt.imshow(mm[2] ,origin='lower', cmap='gray')
                # plt.subplot(4,1,4)
                # plt.imshow(mm[3] ,origin='lower', cmap='gray')
                # plt.draw()
                # plt.pause(0.02)

                mm = mm_.copy(); ll = ll_.copy(); ss = ss_.copy()
                # policy logic settings ...............................
                if evaluate:
                    policy = "mpa2"
                else:
                    
                    if sum(collision_que) >= 2:
                        policy = "collision"
                    else:
                        # policy ="mpa2"
                        if stack_ctr <= 100:
                            policy = "random"
                            # policy = "mpa"
                        else:
                            check_policy =(step_ctr <= agent.batch_size) or (random.random() < (agent.epsilon -agent.epsilon*i/n_games))

                            if check_policy:
                                policy = "random "
                            else:
                                policy = "map2" 
                # action selection ......................................
                cut_step = env.robot.cut_step

                action = agent.choose_action(mm,ll,ss, policy,stack_ctr)
                if action == None:
                    print("Action is None!")
                    action = 1 
                print("policy: ", policy,  " ==> action: ", action)
                # print step info
                if not load_checkpoint:
                    print(" Call to Learn!")
                    agent.learn('global')

                # ********************** Stack Step End ***********************************

            # ...................... Learn .........................

                # actor_loss.append(agent.get_actor_loss())
                # critic_loss.append(agent.get_critic_loss())
        # Back to episode loop
        score_list.append(score)
        if score > best_score:
            best_score = score
            if not load_checkpoint:
                if step_ctr >= 65:
                    agent.save_models()
                    print('GAME Round ' + str(i+1) +" Model Saved !!!")
        save_reward(step_list,reward_list, pack_file +'reward.txt')
        save_reward(interval_list,average_reward, pack_file + 'ave_reward.txt')
        save_reward(interval_list,coverage_list,pack_file + 'coverage_rate.txt')
        save_score(episode_list, score_list, pack_file + 'score.txt')        
            
     
            
 