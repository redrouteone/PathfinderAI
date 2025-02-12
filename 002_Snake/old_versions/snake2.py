import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from collections import deque

# -------------------------
# Hyperparameters / Config
# -------------------------
WIDTH = 400
HEIGHT = 400
BLOCK_SIZE = 20
FPS = 60   # Frames per second for rendering

NUM_EPISODES = 10000   # How many games (episodes) to train
MAX_STEPS_PER_EPISODE = 1000
RENDER_EVERY = 50   # Render one episode every X (to speed training)

GAMMA = 0.9
LR = 0.001
MEMORY_SIZE = 100_000
BATCH_SIZE = 64
EPSILON_START = 1.0000
EPSILON_END = 0.01
EPSILON_DECAY = 0.999

# Reward settings
REWARD_APPLE = 50.0
REWARD_DEATH = -10.0
REWARD_STEP = -0.01

# Actions: [0: straight, 1: turn_left, 2: turn_right]
ACTIONS = [0, 1, 2]

# Grid dimensions
GRID_WIDTH = WIDTH // BLOCK_SIZE    # e.g., 20
GRID_HEIGHT = HEIGHT // BLOCK_SIZE  # e.g., 20

# -------------------------
# Colours for Rendering
# -------------------------
SNAKE_BODY_COLOR = (0, 204, 0)       # Classic green
SNAKE_HEAD_COLOR = (0, 51, 0)     # Cyan-ish (different from the body)
APPLE_COLOR = (255, 0, 0)            # Red

# -------------------------
# Snake Game Environment (Full Grid Version)
# -------------------------
class SnakeGame2:
    def __init__(self, w=WIDTH, h=HEIGHT, block_size=BLOCK_SIZE):
        self.w = w
        self.h = h
        self.block_size = block_size
        self.grid_width = w // block_size
        self.grid_height = h // block_size
        
        # Initialize PyGame for rendering
        pygame.init()
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("RL Snake 2")
        self.clock = pygame.time.Clock()
        
        self.reset()
    
    def reset(self):
        # Use grid coordinates.
        # The snake is represented as a list of (x, y) grid positions.
        self.snake = [(self.grid_width // 2, self.grid_height // 2)]
        # Start moving right
        self.direction = (1, 0)
        self.score = 0
        self.apple = self._place_apple()
        return self._get_state()
    
    def step(self, action):
        """
        Action: 0 = straight, 1 = turn left, 2 = turn right
        """
        self._change_direction(action)
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        
        reward = REWARD_STEP
        done = False
        
        # Check for collision with walls
        if (new_head[0] < 0 or new_head[0] >= self.grid_width or
            new_head[1] < 0 or new_head[1] >= self.grid_height):
            reward += REWARD_DEATH
            done = True
            return self._get_state(), reward, done
        
        # Check for collision with self
        if new_head in self.snake:
            reward += REWARD_DEATH
            done = True
            return self._get_state(), reward, done
        
        # Move snake: insert new head
        self.snake.insert(0, new_head)
        
        # Check if apple is eaten
        if new_head == self.apple:
            self.score += 1
            reward += REWARD_APPLE
            self.apple = self._place_apple()
        else:
            # Remove tail if no apple eaten
            self.snake.pop()
        
        return self._get_state(), reward, done
    
    def render(self):
        self.screen.fill((0, 0, 0))
        
        # Draw apple
        apple_x, apple_y = self.apple
        pygame.draw.rect(self.screen, APPLE_COLOR,
                         (apple_x * self.block_size, apple_y * self.block_size,
                          self.block_size, self.block_size))
        
        # Draw snake: head in a different colour than body
        for i, (x, y) in enumerate(self.snake):
            color = SNAKE_HEAD_COLOR if i == 0 else SNAKE_BODY_COLOR
            pygame.draw.rect(self.screen, color,
                             (x * self.block_size, y * self.block_size,
                              self.block_size, self.block_size))
        
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def _place_apple(self):
        # Randomly place an apple on the grid, avoiding the snake's body.
        while True:
            pos = (random.randint(0, self.grid_width - 1),
                   random.randint(0, self.grid_height - 1))
            if pos not in self.snake:
                return pos
    
    def _get_state(self):
        """
        Returns a full grid representation of the board as a 3-channel array:
          - Channel 0: snake body (excluding the head)
          - Channel 1: snake head
          - Channel 2: apple
        The array shape is (3, grid_height, grid_width) with float values.
        """
        state = np.zeros((3, self.grid_height, self.grid_width), dtype=np.float32)
        # Mark snake body (all segments except head) on channel 0
        for pos in self.snake[1:]:
            x, y = pos
            state[0, y, x] = 1.0
        # Mark snake head on channel 1
        head_x, head_y = self.snake[0]
        state[1, head_y, head_x] = 1.0
        # Mark apple on channel 2
        apple_x, apple_y = self.apple
        state[2, apple_y, apple_x] = 1.0
        return state
    
    def _change_direction(self, action):
        """
        Change the direction of movement based on the current direction.
        For grid movement:
          - Left turn: (dx, dy) becomes (-dy, dx)
          - Right turn: (dx, dy) becomes (dy, -dx)
          - Straight: no change.
        """
        dx, dy = self.direction
        if action == 1:  # turn left
            self.direction = (-dy, dx)
        elif action == 2:  # turn right
            self.direction = (dy, -dx)
        # If action == 0, keep the same direction.

# -------------------------
# Convolutional DQN for Grid-Based State
# -------------------------
class DQN2(nn.Module):
    def __init__(self, action_dim, grid_height=GRID_HEIGHT, grid_width=GRID_WIDTH):
        super(DQN2, self).__init__()
        # The input has 3 channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        # After two pooling layers, dimensions are reduced by a factor of 4.
        conv_height = grid_height // 4
        conv_width = grid_width // 4
        conv_out_size = 64 * conv_height * conv_width
        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        # x should have shape: (batch, 3, grid_height, grid_width)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------
# Replay Memory
# -------------------------
class ReplayMemory:
    def __init__(self, capacity=MEMORY_SIZE):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# -------------------------
# Training Loop with Checkpointing
# -------------------------
def train():
    env = SnakeGame2()
    action_dim = len(ACTIONS)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    policy_net = DQN2(action_dim).to(device)
    target_net = DQN2(action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)
    
    epsilon = EPSILON_START
    episode_rewards = []
    high_score = -float('inf')
    start_episode = 0

    # -------------------------
    # Checkpoint prompt
    # -------------------------
    checkpoint_path = input("Enter checkpoint file path (default 'checkpoint.pth'): ").strip()
    if checkpoint_path == "":
        checkpoint_path = "checkpoint.pth"
    
    resume_training = False
    if os.path.exists(checkpoint_path):
        answer = input(f"Checkpoint '{checkpoint_path}' exists. Do you want to resume training from it? (y/n): ").strip().lower()
        if answer.startswith("y"):
            resume_training = True
    else:
        print("No checkpoint found. Starting training from scratch.")
    
    if resume_training:
        print("Loading checkpoint from", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy_net.load_state_dict(checkpoint["policy_net_state"])
        target_net.load_state_dict(checkpoint["target_net_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        epsilon = checkpoint["epsilon"]
        start_episode = checkpoint["episode"] + 1
        episode_rewards = checkpoint.get("episode_rewards", [])
        memory.buffer = checkpoint.get("replay_memory", memory.buffer)
        high_score = checkpoint.get("high_score", -float('inf'))
        print(f"Resuming from episode {start_episode}, epsilon = {epsilon:.4f}")
    else:
        print("Starting training from scratch.")
    
    # -------------------------
    # Training Loop
    # -------------------------
    for episode in range(start_episode, NUM_EPISODES):
        state = env.reset()  # state: (3, grid_height, grid_width)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        total_reward = 0
        
        for t in range(MAX_STEPS_PER_EPISODE):
            if episode % RENDER_EVERY == 0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                env.render()
            
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = torch.argmax(q_values, dim=1).item()
            
            next_state, reward, done = env.step(action)
            total_reward += reward
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Store transition (convert to CPU numpy arrays for storage)
            memory.push(state.squeeze(0).cpu().numpy(),
                        action,
                        reward,
                        next_state_tensor.squeeze(0).cpu().numpy(),
                        done)
            
            state = next_state_tensor
            
            if len(memory) > BATCH_SIZE:
                states_b, actions_b, rewards_b, next_states_b, dones_b = memory.sample(BATCH_SIZE)
                states_b = torch.tensor(np.array(states_b), dtype=torch.float32).to(device)
                actions_b = torch.tensor(actions_b, dtype=torch.long).to(device)
                rewards_b = torch.tensor(rewards_b, dtype=torch.float32).to(device)
                next_states_b = torch.tensor(np.array(next_states_b), dtype=torch.float32).to(device)
                dones_b = torch.tensor(dones_b, dtype=torch.float32).to(device)
                
                q_values = policy_net(states_b)
                q_values = q_values.gather(1, actions_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_net(next_states_b).max(1)[0]
                    target_q = rewards_b + (1 - dones_b) * GAMMA * next_q_values
                
                loss = F.mse_loss(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
        
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            print(f"AVE, {episode+1}, {NUM_EPISODES}, {avg_reward:.2f}, {epsilon:.4f}")
        
        if total_reward > high_score:
            high_score = total_reward
            print(f"HIGH, {episode+1}, {NUM_EPISODES}, {total_reward:.2f}, {epsilon:.4f}")
        
        # Save checkpoint every 100 episodes
        if (episode + 1) % 100 == 0:
            checkpoint = {
                "episode": episode,
                "policy_net_state": policy_net.state_dict(),
                "target_net_state": target_net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epsilon": epsilon,
                "episode_rewards": episode_rewards,
                "high_score": high_score,
                "replay_memory": memory.buffer,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at episode {episode+1}")
    
    pygame.quit()
    print("Training complete!")
    print(f"Final Average Reward (last 10 episodes): {sum(episode_rewards[-10:]) / 10:.2f}")

if __name__ == "__main__":
    train()