<div align="center">

# Ex-RL Virtual Laboratory 🧪

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
<img alt="Streamlit" src="assets/streamlit.jpg" height="30"/>
<img alt="Gym" src="https://img.stackshare.io/service/12581/gym.png" height="35" style="background-color: black;"/>
<img alt="tqdm" src="https://avatars.githubusercontent.com/u/12731565?s=280&v=4" height="30" />

👋 welcome to the official repository of **_[Ex-RL: Experience-Based Reinforcement Learning]()_**

</div>

## 📖 Table of Contents

- [⚙️ Setup](#setup)
- [📄 Pages](#pages)
  - [👷 Pipeline QTable](#pipeline)
  - [🎮 Review Agent](#review)
  - [🕵️ HMM Trainer](#hmm)
  - [🏃‍♂️ Ex-RL](#exrl)
- [👥 Contributors](#contributors)
- [🗣️ Citation](#citation)
- [🤝 Contributing](#contributing)
- [📚 Libraries](#libraries)
- [🔨 Environment Documentation](#doc)
  - [`create.py`](#createpy)
  - [`encoders`](#encoders)
  - [Encoder](#encoder)
  - [`__init__.py`](#init)
  - [🖊️ Register & `EnvSelector`](#register)

<a name="setup"></a>

## ⚙️ Setup

<a name="manual"></a>

```bash
git clone https://github.com/alighandij/Ex-RL-virtual-lab.git 
cd Ex-RL-virtual-lab
python3 -m venv venv
# Activate the virtual environment based on your OS
pip3 install gym && pip3 install -r requirements.txt
streamlit run App.py
```

<a name="pages"></a>

## 📄 Pages

<a name="pipeline"></a>

### 👷 Pipeline QTable

for Training `QTableAgents` with customized environments.

![pipeline](./assets/pipeline.gif)

<a name="review"></a>

### 🎮 Review Agent

To view information about an agent and play it.

![review](assets/review.gif)

<a name="hmm"></a>

### 🕵️ HMM Trainer

For training an HMM based on trained `QTableAgents`.

![hmm](assets/hmm.gif)

<a name="exrl"></a>

### 🏃‍♂️ Ex-RL

To execute an Ex-RL algorithm with a trained HMM in a different environment.

![exrl](assets/exrl.gif)

<a name="contributors"></a>

## 👥 Contributors

The contributors, listed in chronological order, are:

- Ali Ghandi
- Azam Kamranian
- Mahyar Riazati

<a name="citation"></a>

## 🗣️ Citation

```bibtex

```

<a name="contributing"></a>

## 🤝 Contributing

Contributions to this project are welcome. If you have any ideas or suggestions, feel free to open an issue or submit a pull request.

<a name="libraries"></a>

## 📚 Libraries

- [`tqdm/tqdm`](https://github.com/tqdm/tqdm)
- [`Wirg/stqdm`](https://github.com/Wirg/stqdm)
- [`openai/gym`](https://github.com/openai/gym)
- [`numpy/numpy`](https://github.com/numpy/numpy)
- [`scipy/scipy`](https://github.com/scipy/scipy)
- [`mwaskom/seaborn`](https://github.com/mwaskom/seaborn)
- [`hmmlearn/hmmlearn`](https://github.com/hmmlearn/hmmlearn)
- [`pandas-dev/pandas`](https://github.com/pandas-dev/pandas)
- [`streamlit/streamlit`](https://github.com/streamlit/streamlit)
- [`matplotlib/matplotlib`](https://github.com/matplotlib/matplotlib)
- [`mhyrzt/MultiMountains`](https://github.com/mhyrzt/MultiMountains)
- modified version of [`simon-larsson/ballbeam-gym`](https://github.com/simon-larsson/ballbeam-gym)

<a name="doc"></a>

## 🔨 Environment Documentation

To add a new environment to the pipeline, create a folder at `modules/environments/envs` with the following structure:

```text
your_awesome_env
├── create.py
├── reward.py
├── __init__.py
└───encoders
    ├── encoder_1.py
    └─  __init__.py
```

<a name="cretepy"></a>

### `create.py`

This file should contain a function called `create` for creating and customizing the environments.
_**⚠️NOTE**_: Avoid using `gym.create`instead, refer to the source code from OpenAI Gym unless it supports direct customization (as demonstrated in the `cartpole` example).

```python
def create(**kwargs):
    beam_length = kwargs.get("beam_length")
    config = {
        'timestep': 0.05,
        'setpoint': 0.4,
        'beam_length': beam_length,
        'max_angle': 0.2,
        'init_velocity': 0.0,
        'action_mode': 'discrete'
    }
    env = gym.make("BallBeamSetpoint-v0", **config)
    return env
```

_**⚠️NOTE**_: When using gym environments with the latest version of Gymnasium, make sure to wrap them using `GymWrapper` from `exrl.gym_wrapper`.

<a name="encoders"></a>

### `encoders`

Define your state encoders in this folder and then register them in the `__init__.py` file:

```python
from .your_awesome_encoder import your_awesome_encoder

ENCODERS = {
    "Awesome Encoder": your_awesome_encoder
}
```

<a name="encoder"></a>

### Encoder

An encoder is a function that takes two arguments:

1. `samples`, which is a dictionary containing `agent_id` and samples:

```python
samples = {
    "agent_id_1": [
        sample_1,
        sample_2,
        ...,
        sample_n
    ],
    ..., # other agents
    "agent_id_n": [
        sample_1,
        sample_2,
        ...,
        sample_n
    ],
}
```

2. `pool` to obtain additional information about the agent.

**_Example_**:

```python
def encoding_function(samples: dict, pool):
    sequences = []
    for agent_id, values in samples.items():
        for trajectories in values:
            slope = pool.get_agent_info(agent_id).angle_perturbation
            slope = list(slope)[0]
            encoded = encode_trajectory(slope, trajectories)
            sequences.append(encoded)
    return sequences
```

<a name="init"></a>

### `__init__.py`

Describe your environment using the provided structure.

```python
from .create import create
from .encoder import ENCODERS

YOUR_ENV = {
    "Your Awesome Env": {
        "create": create,
        "state_shape": YOUR_ENV_STATE_SHAPE: int,
        "parameters": {
            "param_1": {
                "options": [50, 100, 150, 200, 250],
                "default": 150,
            },
        },
        "agent_configs": {
            # This dictionary is optional. It only affects the parameter selection and makes the process more convenient.
            "count": 20,
            "target": -120
            "discrete": 32,
        },
        "encoders": ENCODERS
    }
}
```

<a name="register"></a>

### 🖊️ Register

After adding your agents, register your environment in `modules/environments/selector.py` as follows:

```python
from .envs.your_awesome_env import YOUR_ENV
ENVIRONMENTS = {
    # OTHER ENVS
    **YOUR_ENV,
}
```

### `EnvSelector`

the `EnvSelector` allows you to effortlessly access environment configurations. This streamlined tool grants you quick access to a variety of settings and parameters associated with different environments.
