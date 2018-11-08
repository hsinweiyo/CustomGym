import numpy as np
from custom_gym.classic_control import MassPointTrajEnv

class MassPointTrajInstrEnv(MassPointTrajEnv):
    
    def reset(self, task=None, num_task=2):
        obs = super().reset(task, num_task)

        # Change instruction
        shuffle_matrix = np.array([[0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1],
                                   [0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0],
                                   [0, 0, 1, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 0]])
        self.instr = shuffle_matrix.dot(self.instr)
        assert self.instr_space.contains(self.instr), "%r (%s) invalid task" % (self.instr, type(self.instr))

        return self.get_obs()
