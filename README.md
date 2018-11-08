# Custom Gym Environment
Custom OpenAI gym environment

## Installation
```
cd CustomGym
pip install -e .
```

## Usage
Just install and import the package ```custom_gym``` and here you go!
```python
import gym
import custom_gym
from custom_gym import RecorderWrapper

# classic_control
env = gym.make('MassPointGoal-v0')

# recorder wrapper
env = RecorderWrapper(env, './test_data/', file_format='json')
```
See detail example in ```test.py```.

## Reference
* [OpenAI Gym](https://github.com/openai)<br>
  Actually this project is following the tutroial of ```gym```.

