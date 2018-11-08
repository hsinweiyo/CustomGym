import gym
import custom_gym
from custom_gym import RecorderWrapper
import numpy as np
import argparse

# parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--env-id', type=str, default='MassPointTraj-v0')
parser.add_argument('--is-record', type=bool, default=True)
parser.add_argument('--is-for-bc', type=bool, default=True)
parser.add_argument('--save-on-finish', type=bool, default=False)
parser.add_argument('--file-path', type=str, default='./dataset/')
parser.add_argument('--file-format', type=str, default='json')
args = parser.parse_args()

env_id = args.env_id
is_record = args.is_record
is_for_bc = args.is_for_bc
save_on_finish = args.save_on_finish
file_path = args.file_path
file_format = args.file_format

# Create Environment
env = gym.make(env_id)
if is_record:
    env = RecorderWrapper(env, file_path, file_format=file_format, save_on_finish=save_on_finish)

# Print action & observation space
print(env.action_space)
print(env.observation_space)

# Target Position
targets = range(18, 180, 36)
targets = [np.deg2rad(x) for x in targets]
targets = np.array([(np.cos(x), np.sin(x)) for x in targets])
targets = np.concatenate(([(0.25, 0)], [(-0.25, 0)], targets))
print(targets)

# parameter
rotate_scale = 0.3
threshold = rotate_scale * 0.01

# Test Environment
for i_episode in range(10):
  # Reset Environment
  obs = env.reset()
  # specify target
  task = np.array([i_episode % 2, 2+i_episode % 5])
  
  if not is_record:
    # w/o recorder wrapper
    obs = env.unwrapped.reset(task) 
  else:
    # w/ recorder wrapper
    obs = env.unwrapped.unwrapped.reset(task)
    # hack expert action for bc
    if is_for_bc:
      env.traj['expert_action'] = []
  
  t = 0
  rewards = 0

  agent = obs[:2]
  instr = obs[-7:]
  print(instr)
  #target = targets[np.where(instr[2] == 1)[0]][0]
  face = np.array([0, 1])
  # Run Episode
  while True:
    # Render Environment
    env.render()
    # Interact with Environment
    action = [1, 0]
    # target direction & delta angle
    target_dir = targets[task[0]] - agent
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

    else:
      action[0] = 1

    expert_action = np.array(action)

    # hack expert action for bc
    if is_record and is_for_bc:
        env.traj['expert_action'].append(expert_action)

    # Random action
    #if t < 10:
    action = env.action_space.sample()
    action = action*0. + expert_action*1

    obs, reward, done, info = env.step(action)
    
    if info['done_status'] == 'Right Target':
        print(rewards)
    rewards += reward
    if info['done_status'] == 'Right Target':
        print(rewards)
        task = task[1:]

    agent = obs[:2]
    face = obs[2:4]

    t = t+1

    # Check Done
    if done:
      print('Reward: {}'.format(rewards))
      print("Episode finished after {} timesteps".format(t+1))
      break

# Close Environment
env.close()
