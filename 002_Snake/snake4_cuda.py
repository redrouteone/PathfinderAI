import os
import numpy as np
import pygame
import random
import time
import torch

# ========================
# === Training Parameters
# ========================
TOTAL_EPISODES               = 2500000   # Total episodes (games)
MAX_MOVES                    = 1000      # Max moves per episode
PRINT_EVERY                  = 1000      # Print stats every N episodes
RENDER_GAME                  = True      # True: show the game window, False: no window

# Show the game window every Nth episode
RENDER_EVERY_EPISODE         = 5000      # (Set to 0 if you never want normal rendering)

# Replay the game ONLY if a new high score is reached?
# - True  -> no normal renderinfg every N episodes; only do a replay if new high score
# - False -> normal rendering logic (every RENDER_EVERY_EPISODE episodes); no high‐score replays
REPLAY_HIGH_SCORE_GAMES_ONLY = False

ONLY_SHOW_HIGH_SCORES        = False     # If you also want to suppress standard stats printing, set True.
MOVING_AVG_WINDOW            = 5         # Window size for moving average of printed stats

# --- Q-Learning + Epsilon ---
GAMMA            = 0.95      # Discount factor
LEARNING_RATE    = 0.1
START_EPSILON    = 1.0
END_EPSILON      = 0.01

# Option 1: multiplicative decay per episode
EPSILON_DECAY          = 0.9995

# Option 2: linear decay portion
DECAY_PORTION          = 0.8    # Fraction of episodes to linearly decay from START->END
EPSILON_DECAY_TYPE     = "linear"  # "multiplicative" or "linear"

# Rewards
APPLE_REWARD   = 10
DEATH_PENALTY  = -10

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SnakeGame:
    def __init__(self, width=1200, height=1200, grid_size=60, render_game=RENDER_GAME):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.render_game = render_game

        # Initialize pygame
        pygame.init()
        if self.render_game:
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption('Snake RL')
            self.clock = pygame.time.Clock()
        else:
            # Minimalist mode so we don't actually pop up a full window
            pygame.display.init()
            pygame.display.set_mode((1, 1))
            self.screen = None
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        # Start snake in the center
        self.snake = [
            (self.width // (2 * self.grid_size)) * self.grid_size,
            (self.height // (2 * self.grid_size)) * self.grid_size
        ]
        self.snake_body = []
        self.direction = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        self.score = 0
        self.moves = 0

        self.place_food()
        return self.get_state()

    def place_food(self):
        while True:
            self.food = [
                random.randint(0, (self.width - self.grid_size) // self.grid_size) * self.grid_size,
                random.randint(0, (self.height - self.grid_size) // self.grid_size) * self.grid_size
            ]
            if self.food not in self.snake_body and self.food != self.snake:
                break

    def get_state(self):
        """
        Returns a small 1D array:
         [danger_straight, dir_up, dir_down, dir_left, dir_right,
          food_up, food_down, food_left, food_right]
        """
        head_x, head_y = self.snake
        food_x, food_y = self.food

        # Danger straight
        danger_straight = False
        if self.direction == 'UP':
            if head_y - self.grid_size < 0 or [head_x, head_y - self.grid_size] in self.snake_body:
                danger_straight = True
        elif self.direction == 'DOWN':
            if head_y + self.grid_size >= self.height or [head_x, head_y + self.grid_size] in self.snake_body:
                danger_straight = True
        elif self.direction == 'LEFT':
            if head_x - self.grid_size < 0 or [head_x - self.grid_size, head_y] in self.snake_body:
                danger_straight = True
        elif self.direction == 'RIGHT':
            if head_x + self.grid_size >= self.width or [head_x + self.grid_size, head_y] in self.snake_body:
                danger_straight = True

        dir_up    = (self.direction == 'UP')
        dir_down  = (self.direction == 'DOWN')
        dir_left  = (self.direction == 'LEFT')
        dir_right = (self.direction == 'RIGHT')

        food_up    = (food_y < head_y)
        food_down  = (food_y > head_y)
        food_left  = (food_x < head_x)
        food_right = (food_x > head_x)

        return np.array([
            danger_straight,
            dir_up, dir_down, dir_left, dir_right,
            food_up, food_down, food_left, food_right
        ], dtype=int)

    def step(self, action):
        """
        Actions:
         0 -> straight
         1 -> turn right
         2 -> turn left
        """
        clock_wise = ['RIGHT', 'DOWN', 'LEFT', 'UP']
        idx = clock_wise.index(self.direction)

        if action == 1:  # turn right
            self.direction = clock_wise[(idx + 1) % 4]
        elif action == 2:  # turn left
            self.direction = clock_wise[(idx - 1) % 4]
        # else: go straight

        self.moves += 1

        # Move snake head
        x, y = self.snake
        if self.direction == 'UP':
            y -= self.grid_size
        elif self.direction == 'DOWN':
            y += self.grid_size
        elif self.direction == 'LEFT':
            x -= self.grid_size
        elif self.direction == 'RIGHT':
            x += self.grid_size
        self.snake = [x, y]

        # Check collisions
        game_over = False
        reward = 0
        if (x < 0 or x >= self.width or
            y < 0 or y >= self.height or
            self.snake in self.snake_body):
            game_over = True
            reward = DEATH_PENALTY
            return self.get_state(), reward, game_over

        # Check food
        if self.snake == self.food:
            self.score += 1
            reward = APPLE_REWARD
            self.snake_body.append(self.food)
            self.place_food()
        else:
            if self.snake_body:
                self.snake_body.pop(0)
            reward = -0.1

        # Add new head to body
        if self.snake_body:
            self.snake_body.append([x, y])

        return self.get_state(), reward, game_over

    def render(self):
        """Draw only if RENDER_GAME=True."""
        if not self.render_game:
            return

        self.screen.fill((0, 0, 0))

        # Snake head
        pygame.draw.rect(self.screen, (0, 255, 0),
                         pygame.Rect(self.snake[0], self.snake[1],
                                     self.grid_size - 2, self.grid_size - 2))

        # Snake body
        for segment in self.snake_body[:-1]:
            pygame.draw.rect(self.screen, (0, 200, 0),
                             pygame.Rect(segment[0], segment[1],
                                         self.grid_size - 2, self.grid_size - 2))

        # Food
        pygame.draw.rect(self.screen, (255, 0, 0),
                         pygame.Rect(self.food[0], self.food[1],
                                     self.grid_size - 2, self.grid_size - 2))

        pygame.display.flip()
        self.clock.tick(30)


class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = GAMMA
        self.learning_rate = LEARNING_RATE
        self.epsilon = START_EPSILON
        self.q_table = {}

    def get_state_string(self, state):
        return ''.join(map(str, state))

    def get_action(self, state):
        state_str = self.get_state_string(state)
        if state_str not in self.q_table:
            self.q_table[state_str] = torch.zeros(self.action_size, device=device)

        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            return torch.argmax(self.q_table[state_str]).item()

    def train(self, state, action, reward, next_state, done):
        # Q-learning update
        state_str = self.get_state_string(state)
        next_state_str = self.get_state_string(next_state)

        if state_str not in self.q_table:
            self.q_table[state_str] = torch.zeros(self.action_size, device=device)
        if next_state_str not in self.q_table:
            self.q_table[next_state_str] = torch.zeros(self.action_size, device=device)

        current_q = self.q_table[state_str][action]
        next_max_q = torch.max(self.q_table[next_state_str])

        new_q = current_q + self.learning_rate * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_str][action] = new_q


def epsilon_update(agent, episode):
    """Decays epsilon once per episode, either multiplicative or linear."""
    if EPSILON_DECAY_TYPE == "multiplicative":
        if agent.epsilon > END_EPSILON:
            agent.epsilon *= EPSILON_DECAY
            if agent.epsilon < END_EPSILON:
                agent.epsilon = END_EPSILON
    elif EPSILON_DECAY_TYPE == "linear":
        decay_cutoff = int(DECAY_PORTION * TOTAL_EPISODES)
        if episode < decay_cutoff:
            fraction = episode / float(decay_cutoff)
            agent.epsilon = START_EPSILON + fraction * (END_EPSILON - START_EPSILON)
        else:
            agent.epsilon = END_EPSILON


def replay_game(env, replay_time=3.0):
    """Replay the last game for `replay_time` seconds, then exit."""
    start_time = time.time()
    replay = True
    while replay:
        # If time is up, break
        if time.time() - start_time >= replay_time:
            break

        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                # Exit replay when any key is pressed
                replay = False

    print("✅ Replay complete, resuming training...")


def train_agent():
    # Create a folder for screenshots
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    screenshot_folder = os.path.join("screenshots", f"game_{timestamp_str}")
    os.makedirs(screenshot_folder, exist_ok=True)

    env = SnakeGame(render_game=RENDER_GAME)
    agent = QLearningAgent(state_size=9, action_size=3)

    scores = []
    rewards = []
    average_scores = []
    average_rewards = []

    # Track best single-episode score so far for screenshots
    best_episode_score_so_far = float('-inf')

    # Track best average score so far (for color highlights)
    best_avg_score = float('-inf')
    prev_avg_score = None

    # ANSI colors
    COLOR_GREEN = "\033[92m"
    COLOR_RED   = "\033[91m"
    COLOR_CYAN  = "\033[96m"
    COLOR_RESET = "\033[0m"

    # Print header (only if we're not hiding everything with ONLY_SHOW_HIGH_SCORES)
    if not ONLY_SHOW_HIGH_SCORES:
        print(f"{'Episode':<10}{'AvgScore':>10}{'AvgReward':>12}"
              f"{'MovAvgScore':>12}{'MovAvgRwd':>12}{'Epsilon':>10}")

    for episode in range(TOTAL_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        # If REPLAY_HIGH_SCORE_GAMES_ONLY is TRUE, we do NOT render every Nth episode at all.
        # Otherwise, revert to normal do_render logic.
        if REPLAY_HIGH_SCORE_GAMES_ONLY:
            do_render = False
        else:
            do_render = (
                RENDER_GAME and
                RENDER_EVERY_EPISODE > 0 and
                (episode + 1) % RENDER_EVERY_EPISODE == 0
            )

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            if env.moves >= MAX_MOVES and not done:
                done = True
                reward = -5  # small penalty for timing out

            agent.train(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

            if do_render:
                env.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

        # Episode done
        scores.append(env.score)
        rewards.append(total_reward)

        # Check if we have a new single-episode high score
        if env.score > best_episode_score_so_far:
            best_episode_score_so_far = env.score

            # Save a screenshot of the final state
            if RENDER_GAME and env.screen is not None:
                env.render()
                screenshot_filename = f"Score_{env.score:05d}_Game_{episode+1}.png"
                screenshot_path = os.path.join(screenshot_folder, screenshot_filename)
                pygame.image.save(env.screen, screenshot_path)
            else:
                screenshot_filename = "NONE"

            # If replay mode is ON, replay the high score game
            if REPLAY_HIGH_SCORE_GAMES_ONLY:
                print(f"🎉 New High Score! {env.score} points in episode {episode+1}. Screenshot saved: {screenshot_filename}")
                print(f"🔄 Replaying high score game: Episode {episode+1} - Score {env.score}")
                replay_game(env)
            else:
                # Otherwise, just mention the new high score
                print(f"New High Score ({env.score})! Saved screenshot: {screenshot_filename}")

        # Decay epsilon
        epsilon_update(agent, episode + 1)

        # Print stats every PRINT_EVERY episodes (only if not in high-score-only mode)
        if not ONLY_SHOW_HIGH_SCORES and (episode + 1) % PRINT_EVERY == 0:
            recent_scores   = scores[-PRINT_EVERY:]
            recent_rewards  = rewards[-PRINT_EVERY:]
            avg_score  = sum(recent_scores) / len(recent_scores)
            avg_reward = sum(recent_rewards) / len(recent_rewards)

            average_scores.append(avg_score)
            average_rewards.append(avg_reward)

            window_scores  = average_scores[-MOVING_AVG_WINDOW:]
            window_rewards = average_rewards[-MOVING_AVG_WINDOW:]
            mov_avg_score  = sum(window_scores) / len(window_scores)
            mov_avg_rwd    = sum(window_rewards) / len(window_rewards)

            color = COLOR_RESET
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                color = COLOR_CYAN
            elif prev_avg_score is not None:
                if avg_score > prev_avg_score:
                    color = COLOR_GREEN
                elif avg_score < prev_avg_score:
                    color = COLOR_RED
            prev_avg_score = avg_score

            print(f"{color}{(episode+1):<10}{avg_score:10.2f}{avg_reward:12.2f}"
                  f"{mov_avg_score:12.2f}{mov_avg_rwd:12.2f}{agent.epsilon:10.4f}"
                  f"{COLOR_RESET}")


if __name__ == "__main__":
    train_agent()