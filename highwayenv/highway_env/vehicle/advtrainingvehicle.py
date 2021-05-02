from typing import List, Tuple, Union

import numpy as np
import copy
from highway_env import utils
#from highway_env.road.road import Road, LaneIndex, Route
from highway_env.types import Vector

from highway_env.vehicle.controller import MDPVehicle


class AdvTrainingVehicle(MDPVehicle):
    
    def __init__(self,
                 road: Road,
                 position: List[float],
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: LaneIndex = None,
                 target_speed: float = None,
                 route: Route = None) -> None:
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.speed_index = self.speed_to_index(self.target_speed)
        self.target_speed = self.index_to_speed(self.speed_index)
        self.training_status = False
        self.policy = None
        self.observation_type = None

    def freezeTraining(self):
        self.training_status = False
        
    def enableTraining(self):
        self.training_status = True

    def set_observation_type(self, observation_type):
        self.observation_type = observation_type

    def set_policy(self,policy):
        self.policy = policy

    def get_policy(self):
        return self.get_policy
    

    @property
    def is_training(self):
        return self.training_status


    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action.
        - 
        - If the action is a speed change, choose speed from the allowed discrete range.
        - Else, forward action to the ControlledVehicle handler.

        :param action: a high-level action
        """

        if (action is None and not self.training_status and self.policy is not None):
            obv = self.observation_type.observe()
            action = self.policy.predict(obv)

        if action == "FASTER":
            self.speed_index = self.speed_to_index(self.speed) + 1
        elif action == "SLOWER":
            self.speed_index = self.speed_to_index(self.speed) - 1
        else:
            super().act(action)
            return
        self.speed_index = int(np.clip(self.speed_index, 0, self.SPEED_COUNT - 1))
        self.target_speed = self.index_to_speed(self.speed_index)
        super().act()

