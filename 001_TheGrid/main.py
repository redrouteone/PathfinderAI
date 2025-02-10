# main.py

import numpy as np
import matplotlib.pyplot as plt
from TheEnvironment import GridWorld
from TheAgent import DQNAgent

# Set up the environment and agent
grid_size = 8
env = GridWorld(grid_size)
state_size = grid_size * grid_size
action_size = 4  # up, right, down, left
agent = DQNAgent(state_size, action_size)

# Training parameters
n_episodes = 500
max_steps = 500
batch_size = 32

# To store reward history of each episode
scores = []

# Training loop
for episode in range(n_episodes):
    # Reset the environment
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    
    for step in range(max_steps):
        # Agent chooses an action
        action = agent.act(state)
        
        # Environment takes the action and returns the new state, reward, and whether the game is done
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        
        # Remember the previous state, action, reward, and done
        agent.remember(state, action, reward, next_state, done)
        
        # Make next_state the new current state for the next frame.
        state = next_state
        
        if done:
            # Print the score and break out of the loop
            print(f"Episode: {episode+1}/{n_episodes}, Score: {step+1}, Epsilon: {agent.epsilon:.2f}")
            scores.append(step+1)
            break
    
    # Train the agent with experiences in memory
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

# Visualize the training progress
plt.plot(scores)
plt.title('Training Progress')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.show()