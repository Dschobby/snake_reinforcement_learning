# Snake reinforcement learning agent
This code provides an environment for the game snake. Using this environment one can train an deep-q network (DQN) agent or a deep-convolutional-q network (DCQN) agent for playing the game using reinforcement learning. Additionally one can choose freely between the following agents:
- User agent: User played game (bump bird with space)
- Random agent: Agent which performs random actions
- DQN agent: Agent using a deep Q-Network for performing actions
- DCQN agent: Agent using a deep convolutional Q-Network for performing actions

The game and its agent is run by initializing with
```python
g = game.Game(agent_name = "dcqn_agent", device = "cuda")
```
To run a game execute the main function
```python
g.main(draw = "True")
```
If using a DQN agent one can train with
```python
g.train_agent(draw = "False", episodes = 200, batches = 100, hyperparameters)
```

## Trained DQN agent example
Example of trained DCQN agent using a training of 200 episodes each with 32 batches of size 64 and learning rate of $\tau = 1e^{-4}$ with hyperparameters $\gamma = 0.9$, $\epsilon_s = 0.9$, $\epsilon_e = 1e^{-2}$. <br /><br />
![](https://github.com/Dschobby/snake_reinforcement_learning/blob/main/animations/snake_animation.gif)

## Required packages
- numpy
- pytorch
- pygame
