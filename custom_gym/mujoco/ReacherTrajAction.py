import numpy as np

from custom_gym.mujoco import ReacherTraj
from custom_gym.mujoco.my_utils import *

class ReacherTrajAction(ReacherTraj):
    
    def step(self, action):
        action = np.array(action)
        action = action * -1
        return super().step(action)
