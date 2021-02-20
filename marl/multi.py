import gym
import highway_env


from rl_agents.agents.common.factory import agent_factory

# Visualisation
import sys
from tqdm.notebook import trange
sys.path.insert(0, './highway-env/scripts/')
import pprint
# Make environment
env = gym.make("highway-v0")
env.configure({
    "controlled_vehicles": 2,
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction"
            # "type": "ContinuousAction"
        }
    },
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics"
        }
    },
    'screen_height': 150,
    'screen_width': 300
    ,"vehicles_count": 10
    # ,'lanes_count': 5  
    # ,"absolute" : True,
    # 'duration': 50,
})

obs, done = env.reset(), False
# pprint.pprint(env.config)


agent_config = {
    "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
    "model": {
        "type": "MultiLayerPerceptron",
        "layers": [256, 256]
    },
    # "double": False,
    "gamma": 0.75, #0.8
    "n_steps": 1,
    "batch_size": 32, #32
    "memory_capacity": 15000,
    "target_update": 50,
    "exploration": {
        "method": "EpsilonGreedy",
        "tau": 6000,
        "temperature": 1.0,
        "final_temperature": 0.05
    },
    "loss_function": "l2"
}
agent = agent_factory(env, agent_config)

# Run episode
for step in trange(env.unwrapped.config["duration"], desc="Running..."):
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        env.reset()
        # break

env.close()
