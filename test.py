import gym
import custom_gym
#from gym import wrappers
from custom_gym import RecorderWrapper

# Create Environment
#env = gym.make('MassPointGoal-v1')
#env = gym.make('MassPointGoalInstr-v1')
#env = gym.make('MassPointGoalAction-v0')
#env = gym.make('MassPointTraj-v1')
#env = gym.make('MassPointTrajInstr-v1')
#env = gym.make('MassPointTrajAction-v1')

#env = gym.make('ReacherGoal-v0')
#env = gym.make('ReacherGoalInstr-v0')
#env = gym.make('ReacherGoalAction-v0')
#env = gym.make('ReacherTraj-v0')
#env = gym.make('ReacherTrajInstr-v0')
env = gym.make('ReacherTrajAction-v0')

# Print action & observation space
print(env.action_space)
print(env.observation_space)

# Record video
#env = wrappers.Monitor(env, './Movie', force=True)

# Record trajectory
#env = RecorderWrapper(env, './test_data/', file_format='json')

# Test Environment
for i_episode in range(10):

    # Reset Environment
    obs = env.reset()
    print(obs)
    t = 0

    # Run Episode
    while True:
        # Render Environment
        env.render()
    
        # Interact with Environment
        action = env.action_space.sample()
        #action = [0.1, 0.1]
        obs, reward, done, info = env.step(action)
        #print(obs)
        #print("Reward: {}".format(reward))
        #print(obs)
        #print(info)
        t = t+1

        # Check Done
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

# Close Environment
env.close()
