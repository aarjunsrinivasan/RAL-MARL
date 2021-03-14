import argparse
import collections.abc
import yaml
import json
import os
import ray

import numpy as np
import gym

from pathlib import Path
from ray import tune
from ray.rllib.utils import try_import_torch
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print, DEFAULT_LOGGERS, TBXLogger
from ray.rllib.utils.schedules import PiecewiseSchedule
from ray.rllib.agents.callbacks import DefaultCallbacks
from gym.envs.registration import register
import matplotlib.pyplot as plt
# from .environments.coverage import CoverageEnv
# from .environments.path_planning import PathPlanningEnv
# from .models.adversarial import AdversarialModel
# from .trainers.multiagent_ppo import MultiPPOTrainer

from environments.coverage import CoverageEnv
from environments.path_planning import PathPlanningEnv
from models.adversarial import AdversarialModel
from trainers.multiagent_ppo import MultiPPOTrainer
torch, _ = try_import_torch()


# config_path="./config/path_planning.yaml"
# with open(config_path, "rb") as config_file:
#         config = yaml.load(config_file)


# config['env_config']['n_agents']=[1,3]

DEFAULT_OPTIONS = {
    'world_shape': [12, 12],
    'state_size': 24,
    'max_episode_len': 50,
    "n_agents": [1,2],
    "disabled_teams_step": [True,False],
    "disabled_teams_comms": [True,False],
    'communication_range': 5.0,
    'ensure_connectivity': True,
    'position_mode': 'random', # random or fixed
     'world_mode': 'warehouse',
     'reward_type': 'coop_only',
    'agents': {
        'visibility_distance': 3,
        'relative_coord_frame': True
    }
}

# env_config:
#     world_shape: [12, 12]
#     state_size: 24
#     max_episode_len: 50
#     n_agents: [1, 15]
#     disabled_teams_step: [True, False]
#     disabled_teams_comms: [True, False]
#     communication_range: 5.0
#     ensure_connectivity: True
#     reward_type: coop_only
#     world_mode: warehouse
#     agents:
#         visibility_distance: 0
#         relative_coord_frame: True

env = PathPlanningEnv(DEFAULT_OPTIONS)
# print(env.reset())
# env.render()

for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        env.render()
        plt.ion()
        plt.show()

        # plt.plot(fig)
        # print("run obs is")
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

