🚀 [**New Dataset Released !**](#-download-dataset)

# An improved version of EasyCarla-RL, OpenAI Gym environment built on the CARLA simulator for model based RL.

## Overview

[It's not the final version, and the readme still needs to be updated.]

EasyCarla-MBRL provides a lightweight and easy-to-use Gym-compatible interface for the CARLA simulator, specifically tailored for reinforcement learning (RL) applications. It integrates essential observation components such as LiDAR scans, ego vehicle states, nearby vehicle information, and waypoints(we add the vision message,such as BEV and camera front view). The environment supports safety-aware learning with reward and cost signals, visualization of waypoints, and customizable parameters including traffic settings, number of vehicles, and sensor range. EasyCarla-MBRL is designed to help both researchers and beginners efficiently train and evaluate RL agents without heavy engineering overhead.

<div align="center">

<table>
  <tr>
    <td><img src="assets/part1.gif" width="100%"/></td>
    <td><img src="assets/part2.gif" width="100%"/></td>
    <td><img src="assets/part3.gif" width="100%"/></td>
  </tr>
</table>

</div>

## Installation

Clone the repository:

```bash
git clone https://github.com/nishijieXJ/EasyCarla-MBRL.git
cd EasyCarla-MBRL
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Install EasyCarla-RL as a local Python package:

```bash
pip install -e .
```

Make sure you have a running [CARLA simulator](https://carla.org/) server compatible with your environment.

For detailed installation instructions, please refer to the [official CARLA docs](https://carla.readthedocs.io/en/0.9.13/start_quickstart/)

## Quick Start

Run a simple demo to interact with the environment:

```bash
python easycarla_demo.py
```

This script demonstrates how to:
- Create and reset the environment
- Select random or autopilot actions
- Step through the environment and receive observations, rewards, costs, and done signals

Make sure your CARLA server is running before executing the demo.

## Advanced Example: Evaluation with Diffusion Q-Learning

For a more advanced usage, you can run a pre-trained [Diffusion Q-Learning](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL) agent in the EasyCarla-RL environment:

```bash
cd example
python run_dql_in_carla.py
```

Make sure you have downloaded or prepared a trained model checkpoint under the `example/params_dql/` directory.

This example demonstrates:
- Loading a pre-trained RL agent
- Interacting with EasyCarla-RL for evaluation
- Evaluating the performance of a real RL model on a simulated autonomous driving task
 
### Dataset Structure (HDF5)

Each sample in the dataset includes the following fields:

```
/                         (root)
├── observations          → shape: [N, 307]        # concatenated: ego_state + lane_info + lidar + nearby_vehicles + waypoints
├── actions               → shape: [N, 3]          # [throttle, steer, brake]
├── rewards               → shape: [N]             # scalar reward per step
├── costs                 → shape: [N]             # safety-related cost signal per step
├── done                  → shape: [N]             # 1 if episode ends
├── next_observations     → shape: [N, 307]        # next-step observations, same format as observations
├── info                  → dict containing:
│   ├── is_collision      → shape: [N]             # 1 if collision occurs
│   └── is_off_road       → shape: [N]             # 1 if vehicle leaves the road
```

* `N` is the number of total timesteps across all episodes (\~1.1 million).
* `observations` and `next_observations` are 307-dimensional vectors formed by concatenating:

  * `ego_state` (9) + `lane_info` (2) + `lidar` (240) + `nearby_vehicles` (20) + `waypoints` (36)

### Observation Format

Each observation in the dataset is stored as a **307-dimensional flat vector**, constructed by concatenating several components in the following order:

```python
# Flattening function used during data generation

def flatten_obs(obs_dict):
    return np.concatenate([
        obs_dict['ego_state'],        # 9 dimensions
        obs_dict['lane_info'],        # 2 dimensions
        obs_dict['lidar'],            # 240 dimensions
        obs_dict['nearby_vehicles'],  # 20 dimensions
        obs_dict['waypoints']         # 36 dimensions
    ]).astype(np.float32)  # Total: 307 dimensions
```

This format allows for efficient training of neural networks while preserving critical spatial and semantic information.

### How to Load and Train with HDF5 Dataset？

This example shows how to load the offline dataset and use it in a typical RL training loop. The model here is a placeholder — you can plug in any behavior cloning, Q-learning, or actor-critic model.

```python
import h5py
import torch
import numpy as np

# === Load dataset from HDF5 ===
with h5py.File('easycarla_offline_dataset.hdf5', 'r') as f:
    observations = torch.tensor(f['observations'][:], dtype=torch.float32)
    actions = torch.tensor(f['actions'][:], dtype=torch.float32)
    rewards = torch.tensor(f['rewards'][:], dtype=torch.float32)
    next_observations = torch.tensor(f['next_observations'][:], dtype=torch.float32)
    dones = torch.tensor(f['done'][:], dtype=torch.float32)

# === (Optional) check shape info ===
print("observations:", observations.shape)
print("actions:", actions.shape)

# === Placeholder model example ===
class YourModel(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # define your model here
        pass

    def forward(self, obs):
        # define forward pass
        return None

# === Training setup ===
model = YourModel(obs_dim=observations.shape[1], act_dim=actions.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = torch.nn.MSELoss()

# === Offline RL training loop ===
for epoch in range(1, 11):  # e.g. 10 epochs
    for step in range(100):  # e.g. 100 steps per epoch
        # sample random batch
        idx = np.random.randint(0, len(observations), size=256)
        obs_batch = observations[idx]
        act_batch = actions[idx]
        rew_batch = rewards[idx]
        next_obs_batch = next_observations[idx]
        done_batch = dones[idx]

        # forward, compute loss
        pred = model(obs_batch)  # e.g. predict action or Q-value
        loss = loss_fn(pred, act_batch)  # just an example

        # backward and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch}] Loss: {loss.item():.4f}  # Replace with your own logging or evaluation")
```

## Project Structure

```
EasyCarla-RL/                    
├── easycarla/                 # Main environment module (Python package)
│   ├── envs/                     
│   │   ├── __init__.py           
│   │   └── carla_env.py       # Carla environment wrapper following the Gym API
│   └── __init__.py               
├── example/                   # Advanced example
│   ├── agents/                   
│   ├── params_dql/               
│   ├── utils/                    
│   └── run_dql_in_carla.py    # Script to run a pretrained RL model
├── easycarla_demo.py          # Quick Start demo script (basic Gym-style environment interaction)
├── requirements.txt              
├── setup.py                      
└── README.md                     
```

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Author

Created by [nishijieXJ](https://github.com/nishijieXJ)

## 💓 Acknowledgement

This project is made possible thanks to the following outstanding open-source contributions:

- [CARLA](https://github.com/carla-simulator/carla)
- [gym-carla](https://github.com/cjy1992/gym-carla)
- [EasyCarla-RL](https://github.com/silverwingsbot)

