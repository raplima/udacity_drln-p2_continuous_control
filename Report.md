## Learning Algorithm

The agent is based on [Deep Deterministic Policy Gradient Agents](https://arxiv.org/abs/1509.02971) with simple (random) replay memory. 
The agent minimizes the MSE loss computed between expected and actual rewards using [Adam](https://arxiv.org/abs/1412.6980). 
The weights of the agent's networks are softly updated at every training step. 
Other hyperparameters are given below:

```python
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.90            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
```

The agent's Actor network is relatively simple, composed of two hidden layers with leaky ReLU activation. 
Both layers contain 128 neurons. The output layer uses an tanh activation
The agent's Critic network is slightly deeper, composed of three hidden layers with leaky ReLU activation. 
The first two layer contain 256 neurons, the third one contain 128 neurons. 


## Plot of Rewards
![alt text](./training.pdf "Rewards per episode - the agent receives an average reward (over 100 episodes) of at least +30. ")  
The environment was solved in 241 episodes. Note the _*version 1*_ of the environment is solved. 

## Ideas for Future Work

This project uses [Deep Deterministic Policy Gradient Agents](https://arxiv.org/abs/1509.02971) and it is able to solve the environment in less than 300 episodes. 
The performance might be improved with other policy gradients methods, such as:
* [Trust Region Policy Optimization)](https://arxiv.org/abs/1502.05477)
* [Distributed Distributional Deterministic Policy Gradients (D4PG)](https://arxiv.org/abs/1804.08617)
Check [this paper](https://arxiv.org/abs/1604.06778) for a discussion of Deep Reinforcement Learning for Continuous Control
