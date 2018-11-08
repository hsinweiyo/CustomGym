import numpy as np

from custom_gym.mujoco import ReacherGoal
from custom_gym.mujoco.my_utils import *

class ReacherGoalInstr(ReacherGoal):

    def _set_target(self, target_id=None):
        self.target_id = np.random.randint(5) if target_id == None else target_id
        self.target_one_hot = one_hot(5, self.target_id)
        #print('Current Target: {}'.format(self.target_id))

        # instruction (color code)
        instr_table = np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],
        ])
        self.target_one_hot = instr_table[self.target_id]
