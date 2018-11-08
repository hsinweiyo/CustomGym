import gym
import custom_gym
from custom_gym import RecorderWrapper
import numpy as np

# Create Environment      #TODO: arg
env = gym.make('MassPointGoal-v0')
#env = gym.make('MassPointGoalInstr-v0')
#env = gym.make('MassPointGoalAction-v0')

# Recorder
is_record = True          #TODO: arg
is_for_bc = True          #TODO: arg
save_on_finish = False    #TODO: arg
file_path = './dataset/'  #TODO: arg
file_format = 'json'      #TODO: arg

if is_record:
    env = RecorderWrapper(env, file_path, file_format=file_format, save_on_finish=save_on_finish)

# Print action & observation space
print(env.action_space)
print(env.observation_space)

# Target Position
targets = range(18, 180, 36)
targets = [np.deg2rad(x) for x in targets]
targets = np.array([(np.cos(x), np.sin(x)) for x in targets])
print(targets)

# parameter
episode_per_task = 1      #TODO: arg
noise_rate = 0.6          #TODO: arg
rotate_scale = 0.3
threshold = rotate_scale * 0.01

# Test Environment
for task_id in range(5):

    episode = 0
    while episode < episode_per_task:

        # Reset Environment
        obs = env.reset()
        if not is_record:
            # w/o recorder wrapper
            obs = env.unwrapped.reset(task_id) 
        else:
            # w/ recorder wrapper
            obs = env.unwrapped.unwrapped.reset(task_id)
            # hack expert action for bc
            if is_for_bc:
                env.traj['expert_action'] = []
            
        t = 0
      
        # coords
        agent = obs[:2]
        instr = obs[-5:]
        target = targets[task_id]
        face = np.array([0, 1])
      
        # Run Episode
        while True:
            # Render Environment
            env.render()
            
            # Interact with Environment
            action = [1, 0]
            # target direction & delta angle
            target_dir = target - agent
            cos_theta = np.sum(target_dir * face) / (np.linalg.norm(target_dir)*np.linalg.norm(face))
            cos_theta = np.clip(cos_theta, -1, 1)
            delta_theta = np.arccos(cos_theta)
            
            if delta_theta > threshold:
                # right
                dir_sign = 1
                right_dir = np.array([target_dir[1], -target_dir[0]])
                if np.sum(right_dir * face) < 0:
                    dir_sign = -1
                  
                delta_theta = np.clip(delta_theta, -1, 1)
                action[1] = dir_sign * delta_theta / rotate_scale
                action[1] = np.clip(action[1], -1, 1)
        
            else:
                action[0] = 1
            
            expert_action = np.array(action)
            
            # hack expert action for bc
            if is_record and is_for_bc:
                env.traj['expert_action'].append(expert_action)
        
            # Random action
            action = env.action_space.sample()
            action = action*noise_rate + expert_action*(1-noise_rate)
            
            # Step
            obs, reward, done, info = env.step(action)
    
            agent = obs[:2]
            face = obs[2:4]
            
            t = t+1
        
            # Check Done
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                if (not save_on_finish) or info['done_status'] == 'Finish Task':
                    episode += 1
                break

# Close Environment
env.close()
