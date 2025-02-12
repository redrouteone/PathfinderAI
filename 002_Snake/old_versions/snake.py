import pygame
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# -------------------------
# Hyperparameters / Config
# -------------------------
WIDTH = 400
HEIGHT = 400
BLOCK_SIZE = 20
FPS = 60   # Frames per second for rendering
NUM_EPISODES = 50000   # How many games (episodes) to train
MAX_STEPS_PER_EPISODE = 1000
RENDER_EVERY = 50   # Render one episode every X (to speed training)

GAMMA = 0.9
LR = 0.001
MEMORY_SIZE = 100_000
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Reward settings
REWARD_APPLE = 10.0
REWARD_DEATH = -10.0
REWARD_STEP = -0.1

# Actions: [0: straight, 1: turn_left, 2: turn_right]
ACTIONS = [0, 1, 2]

# -------------------------
# Snake Game Environment
# -------------------------
class SnakeGame:
    def __init__(self, w=WIDTH, h=HEIGHT, block_size=BLOCK_SIZE):
        self.w = w
        self.h = h
        self.block_size = block_size
        self.reset()
        
        # For rendering
        pygame.init()
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("RL Snake")
        self.clock = pygame.time.Clock()
        
    def reset(self):
        # Snake initial position
        self.snake = [(self.w//2, self.h//2)]  # list of (x, y) blocks
        self.direction = (self.block_size, 0)  # moving right initially
        self.score = 0
        
        # Place apple
        self.apple = self._place_apple()
        
        # Return initial observation (state)
        return self._get_state()
    
    def step(self, action):
        """
        Action: 0 = straight, 1 = turn left, 2 = turn right
        """
        self._change_direction(action)
        # Move the snake
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        
        reward = REWARD_STEP
        done = False
        
        # Check collision (with walls or self)
        if (new_head[0] < 0 or new_head[0] >= self.w or
            new_head[1] < 0 or new_head[1] >= self.h or
            new_head in self.snake):
            reward += REWARD_DEATH
            done = True
            return self._get_state(), reward, done
        
        # Insert new head
        self.snake.insert(0, new_head)
        
        # Check if apple eaten
        if new_head == self.apple:
            self.score += 1
            reward += REWARD_APPLE
            self.apple = self._place_apple()
        else:
            # Remove tail
            self.snake.pop()
        
        return self._get_state(), reward, done

    def render(self):
        self.screen.fill((0, 0, 0))
        
        # Draw snake
        for x, y in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), (x, y, self.block_size, self.block_size))
        
        # Draw apple
        ax, ay = self.apple
        pygame.draw.rect(self.screen, (255, 0, 0), (ax, ay, self.block_size, self.block_size))
        
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def _place_apple(self):
        x = random.randint(0, (self.w - self.block_size) // self.block_size) * self.block_size
        y = random.randint(0, (self.h - self.block_size) // self.block_size) * self.block_size
        return (x, y)
    
    def _get_state(self):
        """
        Here we define the state as:
         - snake head (x, y)
         - apple (x, y)
         - direction (dx, dy)
         - Possibly we can add the snake's length or other features

        We'll scale positions so they're within [0, 1] by dividing by width/height
        to keep them in a consistent range for the neural net.
        """
        head_x, head_y = self.snake[0]
        apple_x, apple_y = self.apple
        dx, dy = self.direction
        
        return (
            head_x / self.w,
            head_y / self.h,
            apple_x / self.w,
            apple_y / self.h,
            dx / self.block_size,   # dx, dy in {-1, 0, 1} realistically
            dy / self.block_size
        )
    
    def _change_direction(self, action):
        # (dx, dy) changes depending on left/straight/right relative turn
        dx, dy = self.direction
        
        # left turn
        if action == 1:
            # If direction is (dx, dy), turning left is basically
            # (dy, -dx) in a 2D grid
            self.direction = (dy, -dx)
        # right turn
        elif action == 2:
            # turning right would be (-dy, dx)
            self.direction = (-dy, dx)
        # action == 0 => go straight (no change)


# -------------------------
# Deep Q-Network
# -------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
# Training Loop
# -------------------------
def train():
    env = SnakeGame()
    state_dim = len(env.reset())   # our state is a 6-element tuple
    action_dim = len(ACTIONS)
    
    high_score = -float('inf')  # Initialize high_score to negative infinity
    
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)
    
    epsilon = EPSILON_START
    
    # Stats
    episode_rewards = []
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        total_reward = 0
        steps = 0
        
        # Decide if we render this episode
        do_render = ((episode % RENDER_EVERY) == 0)
        
        for t in range(MAX_STEPS_PER_EPISODE):
            if do_render:
                # Basic event loop for PyGame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                env.render()
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = torch.argmax(q_values, dim=1).item()
            
            # Step in the environment
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            next_state_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            
            # Store in replay buffer
            memory.push(
                state.squeeze().tolist(), 
                action, 
                reward, 
                next_state_t.squeeze().tolist(), 
                done
            )
            
            state = next_state_t
            
            # Learn from memory if we have enough samples
            if len(memory) > BATCH_SIZE:
                # sample a batch
                states_b, actions_b, rewards_b, next_states_b, dones_b = memory.sample(BATCH_SIZE)
                
                states_b = torch.tensor(states_b, dtype=torch.float32)
                actions_b = torch.tensor(actions_b, dtype=torch.long)
                rewards_b = torch.tensor(rewards_b, dtype=torch.float32)
                next_states_b = torch.tensor(next_states_b, dtype=torch.float32)
                dones_b = torch.tensor(dones_b, dtype=torch.float32)
                
                # Compute current Q
                q_values = policy_net(states_b)
                # gather the Q-value for the chosen action
                q_values = q_values.gather(1, actions_b.unsqueeze(1)).squeeze(1)
                
                # Compute next Q
                with torch.no_grad():
                    # Double DQN approach can also be used, but let's keep it simple
                    next_q_values = target_net(next_states_b).max(1)[0]
                    target_q = rewards_b + (1 - dones_b) * GAMMA * next_q_values
                
                # Loss
                loss = F.mse_loss(q_values, target_q)
                
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
        
        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        # Update target net
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # After finishing the episode:
        episode_rewards.append(total_reward)

        # Print stats every 50 episodes
        if (episode+1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10 
            print(f"Episode {episode+1}/{NUM_EPISODES}, Average Score (last 10): {avg_reward:.2f}, Epsilon: {epsilon:.2f}")

        # Check for highest score using our separate high_score variable
        if total_reward > high_score:
            high_score = total_reward
            print("\033[91m==============================================\033[0m")
            print("\033[91m!!! New Record Reached: Score {:.2f} on Episode {} !!!\033[0m".format(total_reward, episode+1))
            print("\033[91m==============================================\033[0m")
    
    # Done training
    pygame.quit()
    # Print final results
    print("Training complete!")
    print(f"Final Average Reward (last 10 episodes): {sum(episode_rewards[-10:]) / 10:.2f}")


if __name__ == "__main__":
    train()