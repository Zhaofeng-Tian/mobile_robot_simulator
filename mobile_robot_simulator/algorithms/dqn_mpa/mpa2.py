import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve, manage_memory, save_loss, save_reward, save_score
import random
import sys
sys.path.append('C:\\Users\\61602\\Desktop\\Coding\\python')
from mobile_robot_simulator.tasks.register import Make_Env
import tensorflow as tf
from collections import deque

if __name__ == '__main__':
    # Global setting
    n_games = 100; n_steps = 10000; 
    manage_memory()
    pack_file = 'C:\\Users\\61602\\Desktop\\Coding\\python\\mobile_robot_simulator\\algorithms\\ddqn\\plots2\\'
    
    # env and agent initialize
    env = Make_Env('Mowing-v1')  # v1 is a discrete env

    print("****************init agent*******************")
    agent = Agent(gamma=0.995,epsilon=0.98, lr = 1e-5,
                input_dims=[env.observation_space.shape[0],env.observation_space.shape[1]], env=env,
                n_actions= env.n_actions,
                state_shape=6,batch_size=64,
                eps_min =0.01, eps_dec=5e-7,
                replace = 1000, chkpt_dir= pack_file + 'models\\')
    print('************* agent successfully init ************')
              
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
    visualize = False
    step_ctr = 0; interval_ctr = 0
    n_games = 100; n_steps = 10000

    # *******************  Trainning  *************************
    for i in range(n_games):
        episode_list.append(i+1)
        # game variables initialize
        m, l, s = env.reset() ; prestate = s.copy()
        done = False; score = 0;
        policy = "None"; ref_step = 0
        random_steps = range(ref_step,ref_step+5)
        que = deque([0*i for i in range(0,9)])
        # ************************  Step  *************************
        while not done:
            step_ctr += 1; interval_check = step_ctr % 30
            step_list.append(step_ctr)
            que.append(s[5]); que.popleft()

            
            # policy logic settings ...............................
            if evaluate:
                policy = "ddqn"
            else:
                if s[5]:
                    policy = "collision"

                else:
                    if step_ctr <= 20000:
                        # policy = "random_explore"
                        policy = "model_predictive"
                    else:
                        check_policy =(step_ctr <= agent.batch_size) or (random.random() < (agent.epsilon -agent.epsilon*i/n_games))
                        # if check_policy and step_ctr not in random_steps:
                        #     ref_step = step_ctr
                        #     random_steps = range(ref_step, ref_step+5)
                        # if step_ctr in random_steps:
                        if check_policy:
                            policy = "model_predictive"
                        else:
                            policy = "ddqn" 
            # action selection ......................................
            cut_step = env.robot.cut_step
            # action = agent.choose_action(step_ctr, n_steps, observation, prestate, 
            #                             state, evaluate, policy = policy, cut_step=cut_step)  
            print("policy: ", policy)
            action = agent.choose_action(m,l,s, prestate, policy)  
            print("action: ", action)  
            # step here .............................................
            action_real = env.set_action(action)
            obs_pack, reward, done, info = env.step(action_real)
            m_, l_, s_ = obs_pack
            agent.store_transition(m, l, s , action, reward,
                                   m_, l_, s_, done)
            
            # update some info
            score += reward
            reward_list.append(reward)
            m = m_
            prestate = s.copy()
            s = s_.copy()
            if interval_check == 0:
                interval_list.append(interval_ctr+1)
                average_reward.append(np.mean(reward_list[-30:]))
                coverage_list.append(env.calc_rate())
                interval_ctr += 1 

            # print step info
            print("Episode[{}/{}] ==> Step[{}/{}] ==> Policy {} ==> Action {} ==> Reward {:.4f}"
            .format(i+1, n_games, step_ctr, n_steps, policy, str(action), reward))   
            
            # ...................... Learn .........................
            if not load_checkpoint:
                agent.learn()
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
            
 
       
            
