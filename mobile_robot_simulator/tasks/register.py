from gym.envs.registration import register
import gym
import sys



def Register_Env(task_env, max_episode_steps = 10000):
    """ Check whether the input task_env in the task list """
    task_list = ['Mowing-v0','Mowing-v1']
    if task_env not in task_list:
        return False
    if task_env == 'Mowing-v0':
        register(
            id = task_env,
            entry_point = 'mobile_robot_simulator.tasks.mowing_env:MowingEnv',
            max_episode_steps = max_episode_steps,)
        from mobile_robot_simulator.tasks.mowing_env import MowingEnv
        return True

    elif task_env == 'Mowing-v1':
        register(id = task_env,
                entry_point = 'mobile_robot_simulator.tasks.mowing_env:MowingEnv_v1',
                max_episode_steps = max_episode_steps)

        from mobile_robot_simulator.tasks.mowing_env import MowingEnv_v1
        return True


def Make_Env(env_name):
    """ If in list, gym make the env """
    env_in_list = Register_Env(task_env=env_name, max_episode_steps= 10000)
    if env_in_list:
        print("Registered Task Env Successfully!")
        env = gym.make(env_name)
    else:
        env = None
        raise Exception("Task Env Not Found, Try Another!!")
    return env

# Test
env = Make_Env('Mowing-v1') # Return a instance of MowingEnv class
print(env.n_actions)


