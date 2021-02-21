import gym
import highway_env
from  gym.spaces.utils import flatdim
from gym.spaces import Box
import numpy as np
from rl_agents.agents.common.factory import agent_factory
from rl_agents.trainer.evaluation import Evaluation
from gym.wrappers import Monitor
from tqdm import trange
import sys

from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
from gym.wrappers import Monitor
from pathlib import Path
import base64


# display = Display(visible=0, size=(1400, 900))
# display.start()


def record_videos(env, path="videos"):
    return Monitor(env, path, force=True, video_callable=lambda episode: True)


def show_videos(path="videos"):
    html = []
    for mp4 in Path(path).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


def capture_intermediate_frames(env):
    env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame


env = gym.make("highway-multi-agent-v0")


# env["other_vehicles_type"] = "highway_env.vehicle.behavior.AggressiveVehicle"


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


obs, done = env.reset(), False

# evaluation = Evaluation(env, agent, num_episodes=30, display_env=False)


evaluation = Evaluation(env, agent, num_episodes=3, recover=True)


# evaluation.train()

evaluation.test()
show_videos(evaluation.run_directory)

# Run episode
# for step in trange(env.unwrapped.config["duration"], desc="Running..."):
#     action = agent.act(obs)
#     print(action)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         env.reset()





# env.config[]


# env["other_vehicles_type"] = "highway_env.vehicle.behavior.IDMVehicle"


# agent_config =  {
#     "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
#     "model": {
#         "type": "GraphConvolutionalNetwork",
#         "layers": [32, 16]
#     },
#     "double": False,
#     "gamma": 0.8,
#     "n_steps": 1,
#     "batch_size": 32,
#     "memory_capacity": 15000,
#     "target_update": 50,
#     "exploration": {
#         "method": "EpsilonGreedy",
#         "tau": 6000,
#         "temperature": 1.0,
#         "final_temperature": 0.05
#     },
#     "loss_function": "l2"
# }
