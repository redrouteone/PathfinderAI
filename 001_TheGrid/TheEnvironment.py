import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size))
        self.agent_pos = None
        self.goal_pos = None
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
    def reset(self):
        # Initialize empty grid
        self.grid = np.zeros((self.size, self.size))
        
        # Place agent randomly
        self.agent_pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
        self.grid[self.agent_pos] = 1  # Agent is represented by 1
        
        # Place goal randomly (not on agent)
        while True:
            self.goal_pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            if self.goal_pos != self.agent_pos:
                self.grid[self.goal_pos] = 2  # Goal is represented by 2
                break
        
        # Place obstacles
        for _ in range(self.size):
            while True:
                obstacle_pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
                if obstacle_pos != self.agent_pos and obstacle_pos != self.goal_pos:
                    self.grid[obstacle_pos] = -1  # Obstacles are represented by -1
                    break
                    
        self.render()
        return self.get_state()
    
    def render(self):
        self.ax.clear()
        
        # Set up colors: [-1:blue, 0:green, 1:red, 2:white] for [obstacle, free, agent, goal]
        cmap = plt.cm.colors.ListedColormap(['blue', 'green', 'red', 'white'])
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        # Draw grid
        self.ax.imshow(self.grid, cmap=cmap, norm=norm)
        
        # Add grid lines
        for i in range(self.size + 1):
            self.ax.axhline(i - 0.5, color='black', linewidth=1)
            self.ax.axvline(i - 0.5, color='black', linewidth=1)
        
        # Add labels
        self.ax.text(self.agent_pos[1], self.agent_pos[0], 'A', ha='center', va='center', color='white')
        self.ax.text(self.goal_pos[1], self.goal_pos[0], 'G', ha='center', va='center', color='black')
        
        # Configure axes
        self.ax.set_xticks(range(self.size))
        self.ax.set_yticks(range(self.size))
        plt.title('Agent (A) seeking Goal (G)')
        plt.pause(0.001) # Pause for a short while to update the plot
        
    def step(self, action):
        # Possible moves: up, right, down, left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        new_pos = tuple(np.add(self.agent_pos, directions[action]))
        
        # Default rewards
        reward = -1  # Base movement penalty
        done = False
        
        # Check if move is valid
        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            if self.grid[new_pos] == -1:  # Hit obstacle
                reward = -10
                done = True
            else:
                # Update agent position
                self.grid[self.agent_pos] = 0
                self.agent_pos = new_pos
                self.grid[self.agent_pos] = 1
                
                # Check if goal reached
                if self.agent_pos == self.goal_pos:
                    reward = 100
                    done = True
        
        self.render()
        return self.get_state(), reward, done
    
    def get_state(self):
        return self.grid.flatten()