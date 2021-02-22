import gym
import highway_env

import os
import shutil
from ray.tune.registry import register_env
import ray
import ray.rllib.agents.ppo as ppo





import numpy as np
from typing import Tuple
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle, ControlledVehicle



import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, CircularLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle

from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG



ray.init(local_mode=True,ignore_reinit_error=True)

class HighwayEnv(AbstractEnv):


## Previous rewards
    RIGHT_LANE_REWARD: float = 0.2
    """The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""

    HIGH_SPEED_REWARD: float = 1
    """The reward received when driving at full speed, linearly mapped to zero for lower speeds."""

    LANE_CHANGE_REWARD: float = 0.5
    """The reward received at each lane change action."""
    # ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}
    @classmethod
    def default_config(self) -> dict:
        config = super().default_config()
        config.update({


            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_vehicle_count": 10,
            "duration": 40,  # [s]
            "initial_spacing": 2,
            "collision_reward": -5,  # The reward received when colliding with a vehicle.
            "offroad_terminal": False


        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])




        

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        for _ in range(self.config["controlled_vehicles"]):
            vehicle = self.action_type.vehicle_class.create_random(self.road, 25, spacing=self.config["initial_spacing"])
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
                self.road.vehicles.append(vehicles_type.create_random(self.road))

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        speed = self.vehicle.speed_index if isinstance(self.vehicle, MDPVehicle) \
            else MDPVehicle.speed_to_index(self.vehicle.speed)
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.RIGHT_LANE_REWARD * lane / (len(neighbours) - 1) \
            + self.HIGH_SPEED_REWARD * speed / (MDPVehicle.SPEED_COUNT - 1)
        reward = utils.lmap(reward,
                          [self.config["collision_reward"], self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward





    def _reward(self, action: int) -> float:
        # Cooperative multi-agent reward
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)


    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
            or self.steps >= self.config["duration"]* self.config["policy_frequency"]or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)


    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"

chkpt_root = "tmp/"

trainer_config = DEFAULT_CONFIG.copy()
trainer_config['num_workers'] = 1
trainer_config["train_batch_size"] = 400
trainer_config["sgd_minibatch_size"] = 64
trainer_config["num_sgd_iter"] = 10

trainer = PPOTrainer(trainer_config, HighwayEnv)
for i in range(2):
    print("Training iteration {}...".format(i))
    result=trainer.train()
    chkpt_file = trainer.save(chkpt_root)
    print(status.format(
            i + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            chkpt_file
            ))


env = gym.make('highway-v0')
state = env.reset()
trainer.restore("tmp/checkpoint_2/checkpoint-2")
for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        env.render()
        print("run obs is")
        print(observation)
        action = trainer.compute_action(state)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()


