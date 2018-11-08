import numpy as np

from custom_gym.mujoco import ReacherTraj
from custom_gym.mujoco.my_utils import *

class ReacherTrajInstr(ReacherTraj):

    def _set_target(self, target_id=None):
        if target_id == None:
            self.target_id = [np.random.randint(2), 2+np.random.randint(5)]
        else:
            self.target_id = target_id

        # instruction (shuffled one-hot)
        shuffle_matrix = np.array([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0]
        ])
        self.instr = one_hot(7, self.target_id)
        self.instr = shuffle_matrix.dot(self.instr)  
        #print('Instruction: {}'.format(self.instr))
