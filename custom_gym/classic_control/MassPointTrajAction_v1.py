import numpy as np
from custom_gym.classic_control import MassPointTrajEnv_v1

class MassPointTrajActionEnv_v1(MassPointTrajEnv_v1):

    def step(self, action):
        # Change Action
        action = np.clip(action, self.low_action, self.high_action)
        action = action * -1

        return super().step(action)

