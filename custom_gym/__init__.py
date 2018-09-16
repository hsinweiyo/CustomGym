from custom_gym.utils.recoder import Recoder
from gym.envs.registration import register

# classic control
register(
    id='MountainCarEx-v0',
    entry_point='custom_gym.classic_control:MountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id='MassPoint-v0',
    entry_point='custom_gym.classic_control:MassPointEnv',
    max_episode_steps=200,

)

register(
    id='FiveTarget-v0',
    entry_point='custom_gym.classic_control:FiveTargetEnv',
    max_episode_steps=200,
)

register(
    id='FiveTarget-v1',
    entry_point='custom_gym.classic_control:FiveTargetEnv_v1',
    max_episode_steps=200,
)

register(
    id='FiveTargetColor-v0',
    entry_point='custom_gym.classic_control:FiveTargetColorEnv',
    max_episode_steps=200,
)

register(
    id='FiveTargetColor-v1',
    entry_point='custom_gym.classic_control:FiveTargetColorV1Env',
    max_episode_steps=200,
)

register(
    id='FiveTargetRandColor-v0',
    entry_point='custom_gym.classic_control:FiveTargetRandColorEnv',
    max_episode_steps=200,
)

register(
    id='FiveTargetRandColor-v2',
    entry_point='custom_gym.classic_control:FiveTargetRandColorEnv_v2',
    max_episode_steps=200,
)

# unity
register(
    id='Kobuki-v0',
    entry_point='custom_gym.unity:KobukiEnv',
    
)

# mujoco

register(
    id='ReacherEx-v0',
    entry_point='custom_gym.mujoco:ReacherEnv',
    max_episode_steps=50,
    reward_threshold=-3.75,
)


