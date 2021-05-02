import gym
import highway_env

env = gym.make('highway-v0')
env.configure({
    "controlled_vehicles": 1,

    'screen_height': 150,
    'screen_width': 300
    ,"vehicles_count": 10

    ,'lanes_count': 4,
    # ,"absolute" : True,
    # 'duration': 50,
    # 'show_trajectories': True,
    'vehicles_count': 4, 
})
print(env.config)
# {'observation': {'type': 'Kinematics'}, 'action': {'type': 'DiscreteMetaAction'}, 'simulation_frequency': 15, 'policy_frequency': 1, 'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle', 'screen_width': 300, 'screen_height': 150, 'centering_position': [0.3, 0.5], 'scaling': 5.5, 'show_trajectories': False, 'render_agent': True, 'offscreen_rendering': False, 'manual_control': False, 'real_time_rendering': False, 'lanes_count': 4, 'vehicles_count': 4, 'controlled_vehicles': 1, 'initial_lane_id': None, 'duration': 40, 'ego_spacing': 2, 'vehicles_density': 1, 'collision_reward': -1, 'reward_speed_range': [20, 30], 'offroad_terminal': False} 

# exit()





for ig   in range(1):
    observation = env.reset()
    for t in range(10):
        env.render()
        print("run obs is")
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()