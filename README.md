# RAL-MARL: Reinforcement Learning for Multi-Agent Autonomous Driving

## Overview

RAL-MARL is a research project focused on **Multi-Agent Reinforcement Learning (MARL)** for autonomous driving scenarios. The project combines the highway-env environment with various reinforcement learning algorithms to train intelligent agents for cooperative and competitive driving tasks.

**This work has been published as a technical report:**
> **"Adversarial Agent Behavior Learning in Autonomous Driving Using Deep Reinforcement Learning"**  
> Arjun Srinivasan, Anubhav Paras, Aniket Bera  
> arXiv: [2508.15207](https://arxiv.org/abs/2508.15207) (2025)

## Project Structure

### Core Components

1. **Highway Environment Integration**
   - Based on the [highway-env](https://github.com/eleurent/highway-env) framework
   - Provides realistic autonomous driving simulation environments
   - Supports both single-agent and multi-agent scenarios

2. **Multi-Agent Scenarios**
   - **Cooperative Driving**: Multiple vehicles working together to achieve common goals
   - **Competitive Driving**: Adversarial training scenarios with conflicting objectives
   - **Mixed Scenarios**: Combination of cooperative and competitive behaviors

3. **Reinforcement Learning Algorithms**
   - **Deep Q-Network (DQN)**: Value-based learning for discrete action spaces
   - **Proximal Policy Optimization (PPO)**: Policy-based learning with attention mechanisms
   - **Adversarial Training**: Multi-agent training with opposing objectives

## Key Features

### 🚗 Multi-Agent Highway Environment
- **Multi-Agent Support**: Control multiple vehicles simultaneously
- **Customizable Scenarios**: Configurable number of lanes, vehicles, and behaviors
- **Attention Mechanisms**: Ego-attention networks for better decision making
- **Adversarial Training**: Support for training against rule-based and learned opponents

### 🧠 Advanced RL Algorithms
- **Attention-Based PPO**: Uses ego-attention networks for improved performance
- **Multi-Agent DQN**: Deep Q-learning adapted for multi-agent scenarios
- **Adversarial Training Pipeline**: Iterative training against increasingly sophisticated opponents

### 🎯 Training Scenarios
- **Highway Driving**: Multi-lane highway with traffic flow
- **Adversarial Training**: Training against aggressive and defensive vehicles
- **Cooperative Navigation**: Multiple agents working together
- **Path Planning**: Custom environments for coverage and path planning tasks
- **Adversarial Behavior Learning**: Learning-based method to derive adversarial behaviors for rule-based agents

## Installation

### Prerequisites
```bash
# Clone and install highway-env
git clone https://github.com/eleurent/highway-env.git
cd highway-env
pip install -e .

# Clone and install rl-agents
git clone https://github.com/eleurent/rl-agents.git
cd rl-agents
pip install -e .
```

### Dependencies
```bash
pip install torch stable-baselines3 ray gym numpy matplotlib
```

## Usage

### Quick Start
```bash
# Test the environment
python sampgym.py

# Train a multi-agent DQN
python multidql.py

# Train with PPO and attention
python highwayenv/trainppoatt.py
```

### Environment Testing
```python
import gym
import highway_env

# Create multi-agent highway environment
env = gym.make('highway-multi-agent-v0')
obs = env.reset()

# Run a simple episode
for t in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break
env.close()
```

## Training Scripts

### 1. Multi-Agent DQN (`multidql.py`)
- Trains Deep Q-Network agents in multi-agent highway scenarios
- Configurable network architecture and hyperparameters
- Supports video recording and evaluation

### 2. PPO with Attention (`highwayenv/trainppoatt.py`)
- Implements PPO with ego-attention networks
- Multi-stage adversarial training pipeline
- Custom feature extractors with attention mechanisms

### 3. Ray RLlib Integration (`gymray.py`)
- Integration with Ray RLlib framework
- Distributed training capabilities
- Custom environment registration

### 4. Custom Environments (`custom.py`)
- Path planning and coverage scenarios
- Multi-agent coordination tasks
- Configurable world environments

## Environment Configurations

### Highway Environment
- **Lanes**: Configurable number of lanes (3-5)
- **Vehicles**: Variable traffic density
- **Actions**: Discrete meta-actions (lane change, speed control)
- **Observations**: Kinematic features of nearby vehicles
- **Rewards**: Speed, lane position, collision avoidance

### Multi-Agent Setup
- **Controlled Vehicles**: Multiple AI-controlled vehicles
- **Other Vehicles**: Rule-based traffic (IDM, Aggressive, Defensive)
- **Cooperative Rewards**: Shared objectives for multiple agents
- **Competitive Scenarios**: Adversarial training setups

## Research Applications

### 1. Autonomous Driving
- Multi-vehicle coordination
- Traffic flow optimization
- Safety-critical decision making
- **Adversarial Behavior Learning**: Learning-based method to derive adversarial behaviors for rule-based agents to cause failure scenarios

### 2. Multi-Agent Systems
- Cooperative behavior learning
- Competitive strategy development
- Emergent social behaviors

### 3. Adversarial Training
- Robust policy development
- Opponent modeling
- Safety and reliability testing
- **Adversarial Behavior Learning**: Learning-based method to derive adversarial behaviors that cause failure scenarios in safety-critical applications

## Key Algorithms

### Attention Mechanisms
- **Ego-Attention Networks**: Focus on relevant vehicles
- **Multi-Head Attention**: Parallel attention processing
- **Graph Convolutional Networks**: Relational reasoning

### Training Strategies
- **Curriculum Learning**: Progressive difficulty increase
- **Adversarial Training**: Training against opponents
- **Multi-Stage Training**: Iterative policy improvement

## Results and Evaluation

The project includes:
- **Video Recordings**: Training and evaluation videos
- **Performance Metrics**: Reward curves and statistics
- **Model Checkpoints**: Saved trained models
- **Evaluation Scripts**: Automated performance assessment

## Contributing

This is a research project focused on:
- Multi-agent reinforcement learning
- Autonomous driving simulation
- Adversarial training methodologies
- Attention-based neural architectures

## Citation

If you use this project in your research, please cite our paper:

```bibtex
@article{srinivasan2025adversarial,
  title={Adversarial Agent Behavior Learning in Autonomous Driving Using Deep Reinforcement Learning},
  author={Srinivasan, Arjun and Paras, Anubhav and Bera, Aniket},
  journal={arXiv preprint arXiv:2508.15207},
  year={2025}
}
```

And the highway-env framework:
```bibtex
@misc{highway-env,
  author = {Leurent, Edouard},
  title = {An Environment for Autonomous Driving Decision-Making},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/eleurent/highway-env}},
}
```

## License

This project builds upon the highway-env framework and follows its licensing terms. Please refer to the original highway-env repository for detailed license information.