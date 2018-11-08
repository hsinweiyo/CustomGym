import numpy as np

from custom_gym.mujoco import ReacherGoal
from custom_gym.mujoco.my_utils import *

class ReacherGoalAction(ReacherGoal):

    def step(self, action):
        action = np.array(action)
        action = action * -1
        return super().step(action)
