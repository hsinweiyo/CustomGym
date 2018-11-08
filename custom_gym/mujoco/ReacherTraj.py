import os
import numpy as np

from gym import utils
from custom_gym.mujoco import mujoco_env
from custom_gym.mujoco.my_utils import *
from custom_gym.utils import Recoder

class ReacherTraj(mujoco_env.MujocoEnv, utils.EzPickle):
    
    def __init__(self):

        # Settings
        self.random_task = True # this should be True when training

        # environment setup, set properties
        self.max_timesteps = 20 # max-timesteps
        self.timesteps = 0

        # episode
        self.episode = 0

        # set recoder, etc.
        self._set_target()  # random init target
        self._call_recorder("reset")
        
        # set mujoco
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher_over_cooked.xml', 2)

    def step(self, a):
        # check stage
        n_task = len(self.target_id)
        stage = 0 if n_task == 2 else 1

        # simulation & after simulation
        self.do_simulation(a, self.frame_skip)
        dist1 = np.linalg.norm(self.get_body_com("fingertip") - self.get_body_com("checkpoint"))
        dist2 = np.linalg.norm(self.get_body_com("fingertip") - self.get_body_com("true_target"))
        dist = dist1 if stage == 0 else dist2 # first or second stage

        # target & reward ( num_task_left = len(self.target_id) ) 
        reward_dist = -dist 
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl + -0.4 * (n_task - 1) # reward_dist + reward_ctrl + task
    
        # check done
        self.timesteps += 1
        reward, done, done_status = self._collision_detection(stage, dist, reward)
        #if done_status != "": print(done_status)

        # recording
        self.a, self.reward, self.done = a, reward, done
        min_dist_cp, min_dist_ft = self._call_recorder("step")
        
        # episode
        if done and done_status == 'Finish Task':
            self.episode = (self.episode + 1) % 10

        # return [ob, reward, done, info]
        return self._get_obs(), reward, done, dict(
            reward_dist=reward_dist, reward_ctrl=reward_ctrl, 
            done_status=done_status, 
            coord=self.get_body_com('fingertip'), 
            dist=dist, min_dist_cp=min_dist_cp, min_dist_ft=min_dist_ft
        )

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self, target_id=None):
        # ordered task
        if not self.random_task:
            target_id = [self.episode // 5, 2 + self.episode % 5]
        # setup 
        self._set_target(target_id) # random init target
        self._call_recorder("reset")
        self.timesteps = 0 # timesteps

        # qpos (16 dim): [(finger_2_angle), (true_target_xy), (t1_xy), (t2_xy), (t3_xy), (t4_xy), (checkpoint), (t5_xy)]
        cp_coord = np.sqrt(0.5)
        qpos = 0.21 * np.array([0, 0, -.1, 0, -.3, 0, -.5, 0, -.7, 0, -.9, 0, cp_coord, cp_coord, cp_coord, -cp_coord])
        
        # set target position
        qpos[12], qpos[self.target_id[0]*2+12] = qpos[self.target_id[0]*2+12], qpos[12]
        qpos[13], qpos[self.target_id[0]*2+13] = qpos[self.target_id[0]*2+13], qpos[13]
        qpos[2], qpos[(self.target_id[1]-2)*2+2] = qpos[(self.target_id[1]-2)*2+2], qpos[2]
        qpos[3], qpos[(self.target_id[1]-2)*2+3] = qpos[(self.target_id[1]-2)*2+3], qpos[3]

        # qvel (16 dim): [(finger), (true_target), (t1), (t2), (t3), (t4), (checkpoint), (t5_xy)]
        qvel = np.zeros(16)
 
        # set state & return
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _set_target(self, target_id=None):
        self.target_id = [np.random.randint(2), 2+np.random.randint(5)] if target_id == None else target_id
        self.instr = one_hot(7, self.target_id)
        #print('Instruction: {}'.format(self.instr))

    def _get_obs(self):
        pos = (self.get_body_com("fingertip") / 0.21)[:2]
        q_vel = self.sim.data.qvel.flat[:2]
        return np.concatenate([pos, q_vel, self.instr])

    def _collision_detection(self, stage, dist, reward):
        if dist < 0.019:
            if stage == 0:
                self.target_id = self.target_id[1:]
                reward += 0.5
                return reward, False, "Right Target"
            else:
                reward += 1
                return reward, True, "Finish Task"
        if self.timesteps >= self.max_timesteps:
            reward += -0.5
            return reward, True, "Times Up"
        return reward, False, ""

    def _call_recorder(self, command):
        if command == "reset":
            self.traj = {"coord": [],}
        if command == "step":
            self.traj['coord'].append(self.get_body_com("fingertip").tolist())
            if self.done:  
                # find dist closest to checkpoint
                ctcp = np.argmin(np.linalg.norm(np.array(self.traj['coord'])-self.get_body_com('checkpoint'), axis=1))
                ctft = ctcp + np.argmin(np.linalg.norm(np.array(self.traj['coord'])[ctcp:]-self.get_body_com('true_target'), axis=1))
                min_dist_cp = np.linalg.norm(np.array(self.traj['coord'][ctcp]-self.get_body_com('checkpoint')))
                min_dist_ft = np.linalg.norm(np.array(self.traj['coord'][ctft]-self.get_body_com('true_target')))
                return min_dist_cp, min_dist_ft
            else:
                return 0, 0
