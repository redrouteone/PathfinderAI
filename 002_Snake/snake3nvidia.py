import os
import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from collections import deque

# -------------------------
# Hyperparameters / Config
# -------------------------
WIDTH = 1600            # Internal resolution width
HEIGHT = 1600           # Internal resolution height
BLOCK_SIZE = 80
FPS = 30               # Frames per second for rendering

NUM_EPISODES = 10000   # Total number of episodes to train
MAX_STEPS_PER_EPISODE = 1000
RENDER_EVERY = 100     # Render one episode every X episodes

GAMMA = 0.9
LR = 0.001
MEMORY_SIZE = 100_000
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995  # Set so that epsilon decays to ~0.01 over 30,000 episodes

# Reward settings
REWARD_APPLE = 40.0
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
# Snake Game Environment (Local 3x3 View Version) with Scaled Rendering and Extra "Apple Smell"
# -------------------------
class SnakeGame3:
    def __init__(self, w=WIDTH, h=HEIGHT, block_size=BLOCK_SIZE, fullscreen=False, window_position=None):
        # If a window_position is provided, set it before initializing pygame.
        if window_position is not None:
            os.environ["SDL_VIDEO_WINDOW_POS"] = f"{window_position[0]},{window_position[1]}"
        pygame.init()
        if fullscreen:
            # Full-screen mode uses the current display resolution.
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.w, self.h = self.screen.get_size()
        else:
            # Use the RESIZABLE flag so you can change the window size.
            self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
            self.w = w
            self.h = h
        pygame.display.set_caption("RL Snake 3")
        self.clock = pygame.time.Clock()
        self.block_size = block_size
        # Create an off-screen game surface at the fixed internal resolution.
        self.game_surface = pygame.Surface((WIDTH, HEIGHT))
        self.grid_width = WIDTH // block_size
        self.grid_height = HEIGHT // block_size
        self.reset()

    def reset(self):
        # Start in the middle of the internal grid (using pixel coordinates of the internal resolution)
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

        # Check collision with walls (using internal resolution)
        if (new_head[0] < 0 or new_head[0] >= WIDTH or
            new_head[1] < 0 or new_head[1] >= HEIGHT):
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
        # Draw on the off-screen game_surface (internal resolution)
        self.game_surface.fill((0, 0, 0))
        # Draw apple on the game_surface
        ax, ay = self.apple
        pygame.draw.rect(self.game_surface, APPLE_COLOR,
                         (ax, ay, self.block_size, self.block_size))
        # Draw snake on the game_surface
        for i, (x, y) in enumerate(self.snake):
            color = SNAKE_HEAD_COLOR if i == 0 else SNAKE_BODY_COLOR
            pygame.draw.rect(self.game_surface, color,
                             (x, y, self.block_size, self.block_size))
        # Scale the game_surface to the current window size
        scaled_surface = pygame.transform.scale(self.game_surface, (self.screen.get_width(), self.screen.get_height()))
        self.screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(FPS)

    def _place_apple(self):
        # Random placement avoiding the snake (using internal grid coordinates)
        while True:
            pos = (random.randint(0, self.grid_width - 1) * self.block_size,
                   random.randint(0, self.grid_height - 1) * self.block_size)
            if pos not in self.snake:
                return pos

    def _get_state(self):
        """
        Returns an augmented state vector.
        First 49 elements: flattened 7x7 local view around the snake's head.
          Encoding for the local view:
            0.0 = empty
            1.0 = wall (if out-of-bounds)
            2.0 = apple
            3.0 = snake body (non-head)
            4.0 = snake head (center cell)
        Last 2 elements: relative position of the apple from the snake's head,
          normalized to the range [-1, 1] (relative_x, relative_y).
        """
        # Calculate the snake head's grid position.
        head_px, head_py = self.snake[0]
        head_x = head_px // self.block_size
        head_y = head_py // self.block_size

        view_size = 7
        offset = view_size // 2  # For 7, offset will be 3.
        local_view = np.zeros((view_size, view_size), dtype=np.float32)

        # Fill in the local view.
        for i in range(-offset, offset + 1):  # rows from -3 to 3
            for j in range(-offset, offset + 1):  # columns from -3 to 3
                grid_x = head_x + j
                grid_y = head_y + i
                view_i = i + offset  # convert from [-3, -2, ..., 3] to [0,...,6]
                view_j = j + offset

                if grid_x < 0 or grid_x >= self.grid_width or grid_y < 0 or grid_y >= self.grid_height:
                    local_view[view_i, view_j] = 1.0  # Mark out-of-bounds as wall.
                else:
                    cell_px = grid_x * self.block_size
                    cell_py = grid_y * self.block_size
                    apple_x = self.apple[0] // self.block_size
                    apple_y = self.apple[1] // self.block_size
                    if grid_x == apple_x and grid_y == apple_y:
                        local_view[view_i, view_j] = 2.0  # Apple present.
                    elif (cell_px, cell_py) in self.snake[1:]:
                        local_view[view_i, view_j] = 3.0  # Snake body.
                    else:
                        local_view[view_i, view_j] = 0.0  # Empty.
        # Force the center cell to represent the snake's head.
        local_view[offset, offset] = 4.0

        # Compute relative position of the apple (in grid coordinates)
        apple_x = self.apple[0] // self.block_size
        apple_y = self.apple[1] // self.block_size
        rel_x = apple_x - head_x
        rel_y = apple_y - head_y

        # Normalize the relative position to [-1, 1]
        norm_rel_x = rel_x / (self.grid_width - 1)
        norm_rel_y = rel_y / (self.grid_height - 1)

        # Concatenate the flattened local view with the extra two features.
        state_vector = np.concatenate([local_view.flatten(), np.array([norm_rel_x, norm_rel_y], dtype=np.float32)])
        return state_vector

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
# Training Loop with Checkpointing, CSV-Friendly Output, and Moving Average
# -------------------------
def train():
    # Create a timestamped folder for screenshots so files don't overwrite between runs.
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    screenshot_folder = f"screenshots_{timestamp}"
    if not os.path.exists(screenshot_folder):
        os.makedirs(screenshot_folder)

    # Set window position (e.g., (100, 100)) and choose full-screen mode if desired.
    env = SnakeGame3(fullscreen=False, window_position=(100, 100))
    # Local view: 9 elements (flattened 3x3) plus 2 extra features = 11.
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

    # Moving average window size for score
    moving_avg_window = 300
    prev_moving_avg = None  # Track previous moving average for comparison

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

            # Store experience (stored as numpy arrays to save memory)
            memory.push(state.squeeze(0).cpu().numpy(),
                        action,
                        reward,
                        next_state_tensor.squeeze(0).cpu().numpy(),
                        done)
            state = next_state_tensor

            if len(memory) > BATCH_SIZE:
                states_b, actions_b, rewards_b, next_states_b, dones_b = memory.sample(BATCH_SIZE)
                states_b = torch.tensor(np.array(states_b), dtype=torch.float32, device=device)
                actions_b = torch.tensor(actions_b, dtype=torch.long, device=device)
                rewards_b = torch.tensor(rewards_b, dtype=torch.float32, device=device)
                next_states_b = torch.tensor(np.array(next_states_b), dtype=torch.float32, device=device)
                dones_b = torch.tensor(dones_b, dtype=torch.float32, device=device)

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

        # Compute the moving average over the last moving_avg_window episodes.
        if len(episode_rewards) >= moving_avg_window:
            moving_avg = sum(episode_rewards[-moving_avg_window:]) / moving_avg_window
        else:
            moving_avg = sum(episode_rewards) / len(episode_rewards)

        if prev_moving_avg is None:
            csv_color = "\033[92m"  # Default to green for the first print.
        else:
            # If the current moving average is strictly greater, use green; otherwise (equal or lower) use red.
            csv_color = "\033[92m" if moving_avg > prev_moving_avg else "\033[91m"
        print(csv_color + f"AVE, {episode+1}, {NUM_EPISODES}, {avg_reward:.2f}, {epsilon:.4f}, MOV, {moving_avg:.2f}" + "\033[0m")
        prev_moving_avg = moving_avg

        # CSV-friendly output every 100 episodes:
        if (episode + 1) % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / 100
            print(csv_color + f"AVE, {episode+1}, {NUM_EPISODES}, {avg_reward:.2f}, {epsilon:.4f}, MOV, {moving_avg:.2f}" + "\033[0m")

        # Print high score message in black.
        if total_reward > high_score:
            high_score = total_reward
            print("\033[30m" + f"HIGH, {episode+1}, {NUM_EPISODES}, {total_reward:.2f}, {epsilon:.4f}, MOV, {moving_avg:.2f}" + "\033[0m")

        # Save checkpoint every 500 episodes.
        if (episode + 1) % 500 == 0:
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

        # Save a screenshot only on episodes that are rendered.
        if episode % RENDER_EVERY == 0:
            screenshot_path = os.path.join(screenshot_folder, f"final_state_episode_{episode+1}.png")
            pygame.image.save(env.screen, screenshot_path)

    pygame.quit()
    print("Training complete!")
    print(f"Final Average Reward (last 10 episodes): {sum(episode_rewards[-10:]) / 10:.2f}")

if __name__ == "__main__":
    train()