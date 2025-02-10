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
FPS = 60  # Frames per second for rendering

NUM_EPISODES = 10000  # Total number of episodes to train
MAX_STEPS_PER_EPISODE = 1000
RENDER_EVERY = 50  # Render one episode every X episodes

GAMMA = 0.9
LR = 0.001
MEMORY_SIZE = 100_000
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.99999

# Reward settings
REWARD_APPLE = 50.0
REWARD_DEATH = -10.0
REWARD_STEP = -0.01

# Actions: 0 = straight, 1 = turn left, 2 = turn right
ACTIONS = [0, 1, 2]

# Grid dimensions (for apple placement and walls)
GRID_WIDTH = WIDTH // BLOCK_SIZE  # e.g., 20
GRID_HEIGHT = HEIGHT // BLOCK_SIZE  # e.g., 20

# -------------------------
# Colours for Rendering
# -------------------------
SNAKE_BODY_COLOR = (0, 204, 0)  # Classic green
SNAKE_HEAD_COLOR = (0, 51, 0)   # Darker green for head
APPLE_COLOR = (255, 0, 0)       # Red

# -------------------------
# Snake Game Environment (Local 3x3 View Version)
# -------------------------
class SnakeGame3:
    def __init__(self, w=WIDTH, h=HEIGHT, block_size=BLOCK_SIZE):
        self.w = w
        self.h = h
        self.block_size = block_size
        self.grid_width = w // block_size
        self.grid_height = h // block_size

        # Initialize PyGame for rendering
        pygame.init()
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("RL Snake 3")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Start in the middle of the board (using pixel coordinates)
        self.snake = [(self.grid_width // 2 * self.block_size, 
                       self.grid_height // 2 * self.block_size)]
        self.direction = (self.block_size, 0)  # Start moving right
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

        # Check collision with walls
        if (new_head[0] < 0 or new_head[0] >= self.w or
            new_head[1] < 0 or new_head[1] >= self.h):
            reward += REWARD_DEATH
            done = True
            return self._get_state(), reward, done

        # Check collision with self
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
            self.snake.pop()

        return self._get_state(), reward, done

    def render(self):
        self.screen.fill((0, 0, 0))
        # Draw apple
        ax, ay = self.apple
        pygame.draw.rect(self.screen, APPLE_COLOR,
                         (ax, ay, self.block_size, self.block_size))
        # Draw snake (head and body)
        for i, (x, y) in enumerate(self.snake):
            color = SNAKE_HEAD_COLOR if i == 0 else SNAKE_BODY_COLOR
            pygame.draw.rect(self.screen, color,
                             (x, y, self.block_size, self.block_size))
        pygame.display.flip()
        self.clock.tick(FPS)

    def _place_apple(self):
        # Random placement avoiding the snake
        while True:
            pos = (random.randint(0, self.grid_width - 1) * self.block_size,
                   random.randint(0, self.grid_height - 1) * self.block_size)
            if pos not in self.snake:
                return pos

    def _get_state(self):
        """
        Returns a flattened 3x3 local view around the snake's head.
        Encoding:
          0.0 = empty
          1.0 = wall (if out-of-bounds)
          2.0 = apple
          3.0 = snake body (non-head)
          4.0 = snake head (center cell)
        """
        head_px, head_py = self.snake[0]
        head_x = head_px // self.block_size
        head_y = head_py // self.block_size

        local_view = np.zeros((3, 3), dtype=np.float32)
        for i in range(-1, 2):  # rows
            for j in range(-1, 2):  # columns
                grid_x = head_x + j
                grid_y = head_y + i
                view_i = i + 1  # convert [-1,0,1] to [0,1,2]
                view_j = j + 1

                if grid_x < 0 or grid_x >= self.grid_width or grid_y < 0 or grid_y >= self.grid_height:
                    local_view[view_i, view_j] = 1.0  # wall
                else:
                    cell_px = grid_x * self.block_size
                    cell_py = grid_y * self.block_size
                    apple_x = self.apple[0] // self.block_size
                    apple_y = self.apple[1] // self.block_size
                    if grid_x == apple_x and grid_y == apple_y:
                        local_view[view_i, view_j] = 2.0  # apple
                    elif (cell_px, cell_py) in self.snake[1:]:
                        local_view[view_i, view_j] = 3.0  # snake body
                    else:
                        local_view[view_i, view_j] = 0.0  # empty
        local_view[1, 1] = 4.0  # Center is always the head.
        return local_view.flatten()

    def _change_direction(self, action):
        dx, dy = self.direction
        if action == 1:  # turn left: (dx, dy) -> (dy, -dx)
            self.direction = (dy, -dx)
        elif action == 2:  # turn right: (dx, dy) -> (-dy, dx)
            self.direction = (-dy, dx)
        # action == 0: keep the same direction

# -------------------------
# Deep Q-Network (Simple Fully Connected)
# -------------------------
class DQN3(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN3, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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
# Training Loop with Checkpointing & CSV-Friendly Output
# -------------------------
def train():
    env = SnakeGame3()
    # Local view: 9 elements (flattened 3x3)
    state_dim = len(env.reset())
    action_dim = len(ACTIONS)
    high_score = -float('inf')
    
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN3(state_dim, action_dim).to(device)
    target_net = DQN3(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)
    
    epsilon = EPSILON_START
    episode_rewards = []
    start_episode = 0

    # --- Checkpoint prompt ---
    checkpoint_path = input("Enter checkpoint file path (default 'checkpoint.pth'): ").strip()
    if checkpoint_path == "":
        checkpoint_path = "checkpoint.pth"

    resume_training = False
    if os.path.exists(checkpoint_path):
        answer = input(f"Checkpoint '{checkpoint_path}' exists. Resume training? (y/n): ").strip().lower()
        if answer.startswith("y"):
            resume_training = True
    else:
        print("No checkpoint found. Starting from scratch.")

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

    # --- Training Loop ---
    for episode in range(start_episode, NUM_EPISODES):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0

        for t in range(MAX_STEPS_PER_EPISODE):
            if episode % RENDER_EVERY == 0:
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

            next_state, reward, done = env.step(action)
            total_reward += reward
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            # Store experience (store as numpy arrays to save memory)
            memory.push(state.squeeze(0).cpu().numpy(),
                        action,
                        reward,
                        next_state_tensor.squeeze(0).cpu().numpy(),
                        done)
            state = next_state_tensor

            if len(memory) > BATCH_SIZE:
                # Vectorized conversion of replay batch
                states_b = torch.tensor(np.array(memory.sample(BATCH_SIZE)[0]), dtype=torch.float32, device=device)
                actions_b = torch.tensor(memory.sample(BATCH_SIZE)[1], dtype=torch.long, device=device)
                rewards_b = torch.tensor(memory.sample(BATCH_SIZE)[2], dtype=torch.float32, device=device)
                next_states_b = torch.tensor(np.array(memory.sample(BATCH_SIZE)[3]), dtype=torch.float32, device=device)
                dones_b = torch.tensor(memory.sample(BATCH_SIZE)[4], dtype=torch.float32, device=device)

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
                # Optionally, save a screenshot of the final state:
                pygame.image.save(env.screen, f"final_state_episode_{episode+1}.png")
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            print(f"AVE, {episode+1}, {NUM_EPISODES}, {avg_reward:.2f}, {epsilon:.4f}")

        if total_reward > high_score:
            high_score = total_reward
            # Print high scores in red for emphasis.
            print("\033[91m" + f"HIGH, {episode+1}, {NUM_EPISODES}, {total_reward:.2f}, {epsilon:.4f}" + "\033[0m")

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