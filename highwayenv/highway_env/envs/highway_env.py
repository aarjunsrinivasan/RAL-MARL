import numpy as np
from typing import Tuple
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.controller import MDPVehicle, AdvTrainingVehicle


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    RIGHT_LANE_REWARD: float = 0.1
    """The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""

    HIGH_SPEED_REWARD: float = 0.4
    """The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"]."""

    LANE_CHANGE_REWARD: float = 0
    """The reward received at each lane change action."""

    @classmethod
    def default_config(cls) -> dict:
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
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            controlled_vehicle = self.action_type.vehicle_class.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            for _ in range(others):
                self.road.vehicles.append(
                    other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                )

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.RIGHT_LANE_REWARD * lane / max(len(neighbours) - 1, 1) \
            + self.HIGH_SPEED_REWARD * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                          [self.config["collision_reward"], self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)

class HighwayEnvAdv(HighwayEnv):

  COLLISION_REWARD = -1
  """ The reward received when colliding with a vehicle."""
  RIGHT_LANE_REWARD = 0.1
  """ The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""
  HIGH_VELOCITY_REWARD = 0.4
  """ The reward received when driving at full speed, linearly mapped to zero for lower speeds."""
  LANE_CHANGE_REWARD = -0
  """ The reward received at each lane change action."""

  DEFAULT_CONFIG = {
    "observation": {
        "type": "Kinematics"
    },
    "policy_frequency": 1,  # [Hz]
    "initial_spacing": 2,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,
    "screen_height": 150,
    "centering_position": [0.3, 0.5],
    "collision_reward": COLLISION_REWARD,
    "lanes_count": 4,
    "vehicles_count": 5,
    "duration": 20
  }

  def __init__(self):
    config = self.DEFAULT_CONFIG.copy()
    super(HighwayEnvAdv, self).__init__(config)
    self.steps = 0
    self.reset()

  def reset(self):
    self._create_road()
    self._create_vehicles()
    self.steps = 0
    return super(HighwayEnvAdv, self).reset()

  def step(self, action):
    self.steps += 1
    return super(HighwayEnvAdv, self).step(action)
  

  def _create_vehicles(self):
    """
        Create some new random vehicles of a given type, and add them on the road.
    """
    self.vehicle = AdvTrainingVehicle.create_random(self.road, 25, spacing=self.config["initial_spacing"]) # or ControlledVehicle?
    self.road.vehicles.append(self.vehicle)

    self.adv_vehicles = []

    vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
    for _ in range(self.config["vehicles_count"]):
      adv_vehicle = vehicles_type.create_random(self.road)
      self.adv_vehicles.append(adv_vehicle)
      self.road.vehicles.append(adv_vehicle)
    
  



register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-adv-v0',
    entry_point='highway_env.envs:HighwayEnvAdv',
)
