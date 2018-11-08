import numpy as np
from custom_gym.classic_control import MassPointGoalEnv

class MassPointGoalInstrEnv(MassPointGoalEnv):
    
    def reset(self, task=None):
        obs = super().reset(task)

        # Change instruction
        instr_table = np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],
        ])

        self.instr = instr_table[self.task]
        assert self.instr_space.contains(self.instr), "%r (%s) invalid task" % (self.instr, type(self.instr))

        return self.get_obs()
