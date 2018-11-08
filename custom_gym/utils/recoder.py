import numpy as np
import json

class Recoder:
    
    def __init__(self, file_path):
        print('The Recoder will be deprecated. Use RecorderWrapper instead.')
        
        self.file_path = file_path
        self.count = 0

        self.reset_traj()

    def reset_traj(self):
        self.traj = {
            'action': [],
            'state': []
        }

    def step(self, state=None, action=None):
        if state is not None:
            if type(state) is np.ndarray:
                state = state.tolist()
            self.traj['state'].append(state)

        if action is not None:
            if type(action) is np.ndarray:
                action = action.tolist()
            self.traj['action'].append(action)
    
    def save(self, state=None, action=None):
        # last step
        self.step(state, action)
        
        # save file
        if len(self.traj['state']) > 10:
            with open(self.file_path+'traj_{}.json'.format(self.count), 'w') as fp:
                json.dump(self.traj, fp, indent=2)
            print('Save file: ' + self.file_path + 'traj_{}.json'.format(self.count))

            self.count += 1

        # reset
        self.reset_traj()


