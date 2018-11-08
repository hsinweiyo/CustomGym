import os
import numpy as np

from gym import utils
from custom_gym.mujoco import mujoco_env
from custom_gym.mujoco.my_utils import *
from custom_gym.utils import Recoder

class ReacherGoal(mujoco_env.MujocoEnv, utils.EzPickle):
    
    def __init__(self):
        # Settings
        self.random_task = True # this should be True when training

        # set properties
        self.max_timesteps = 20
        self.timesteps = 0
        
        # set init target
        self._set_target()
 
        # episode
        self.episode = 0

        # set mujoco
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher_five_target_v1.xml', 2)

    def step(self, a):
        # simulation & after simulation
        self.do_simulation(a, self.frame_skip)
        vec = self.get_body_com("fingertip") - self.get_body_com("true_target")
        dist = np.linalg.norm(vec)
        
        # reward
        reward_dist = - dist
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl

        # check done
        self.timesteps += 1
        reward, done, done_status = self._collision_detection(dist, reward)
        #if done_status != "": print(done_status)

        # episode
        if done and done_status == 'Finish Task':
            self.episode = (self.episode + 1) % 5

        # return [ob, reward, done, info]
        return self._get_obs(), reward, done, dict(
            reward_dist=reward_dist, reward_ctrl=reward_ctrl,
            done_status=done_status, coord=self.get_body_com('fingertip'), dist=dist
        )

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self, target_id=None):
        # ordered task
        if not self.random_task:
            target_id = self.episode
        # random init target
        self._set_target(target_id)
        self.timesteps = 0

        # qpos (12 dim): [(finger_2_angle), (true_target_xy), (t1_xy), (t2_xy), (t3_xy), (t4_xy)]
        qpos = np.array([0, 0, 0, .15, -.1, .1, -.2, 0, -.1, -.1, 0, -.15])
        
        # set target position
        qpos[2], qpos[self.target_id*2+2] = qpos[self.target_id*2+2], qpos[2]
        qpos[3], qpos[self.target_id*2+3] = qpos[self.target_id*2+3], qpos[3]

        # qvel (12 dim): [(finger), (true_target), (t1), (t2), (t3), (t4)]
        qvel = np.zeros(12)
        
        # set state & return
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _set_target(self, target_id=None):
        self.target_id = np.random.randint(5) if target_id == None else target_id
        self.target_one_hot = one_hot(5, self.target_id)
        #print('Current Target: {}'.format(self.target_id))

    def _get_obs(self):
        pos = self.get_body_com("fingertip") / 0.21
        return np.concatenate([
            pos[:2],
            self.sim.data.qvel.flat[:2],
            self.target_one_hot
        ])

    def _collision_detection(self, dist, reward):
        if dist < 0.019:
            reward += 1
            return reward, True, 'Finish Task'
        if self.timesteps >= self.max_timesteps:
            reward += -0.5
            return reward, True, 'Times Up'
        return reward, False, ''
