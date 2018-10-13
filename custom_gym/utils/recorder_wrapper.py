import numpy as np
import pickle
import json
import os

class RecorderWrapper:

    def __init__(self, env, file_path, file_format='pickle', jsonify=False, len_threshold=0, save_on_finish=False):
        '''
        file_format: only pickle and json currently
        save_on_finish: save file only when finish the task
        '''

        # gym environmant
        self.unwrapped = env

        # file save path
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)

        # file format
        self.file_format = file_format
        if self.file_format == 'json':
            self.is_jsonify = True
        else:
            self.is_jsonify = jsonify
        # default save as pickle
        if self.file_format != 'json' and self.file_format != 'pickle':
            print('Warning: not support file format "'+file_format+'"')
            self.file_format = 'pickle'

        # parameters
        self.len_threshold = len_threshold
        self.save_on_finish = save_on_finish
        # episode count
        self.count = 0

    def __getattr__(self, attr):
        # called only when attr not found
        return self.unwrapped.__getattribute__(attr)


    def reset_traj(self):
        self.traj = {
            'state': [],
            'action': [],
            'reward': [],
            'info': {}
        }

    def push_pack(self, state, action, reward, info):
        self.push_single('state', state)
        self.push_single('action', action)
        self.push_single('reward', reward)
        self.push_info(info)

    def push_single(self, attr, data):
        # check data
        if data is not None:
            # push data
            self.traj[attr].append(data)

    def push_info(self, info):
        for key in info:
            try:
                self.traj['info'][key].append(info[key])
            except KeyError:
                self.traj['info'][key] = []
                self.traj['info'][key].append(info[key])

    def jsonify(self):
        # convert numpy array to jsonifable object
        for key in self.traj:
            self.traj[key] = np.array(self.traj[key]).tolist()
        for key in self.traj['info']:
            self.traj['info'][key] = np.array(self.traj['info'][key]).tolist()

    def save(self):
        
        # filter out short episode
        if len(self.traj['state']) > self.len_threshold:
            # convert to jsonifable object
            if self.is_jsonify:
                self.jsonify()
            # dump file
            file_name = self.file_path+'traj_{}.'.format(self.count)+self.file_format
            if self.file_format == 'json':
                with open(file_name, 'w') as fp:
                    json.dump(self.traj, fp, indent=2)
            else:
                with open(file_name, 'wb') as fp:
                    pickle.dump(self.traj, fp)

            print('Save file: '+file_name)

            # increase count
            self.count += 1
            # reset trajectory
            self.reset_traj()

    def step(self, action):
        obs, rew, done, info = self.unwrapped.step(action)

        self.push_pack(obs, action, rew, info)
        # save when episode ends
        # TODO additional custom condition 
        if done:
            if (not self.save_on_finish) or info['done_status'] == 'Finish Task':
                self.save()

        return obs, rew, done, info
    
    def reset(self, *args):
        obs = self.unwrapped.reset(*args)

        # reset trajectory
        self.reset_traj()
        # record init state
        self.push_single('state', obs)

        return obs

