# Minotaur's Maze: An MDP Approach to Escape

import numpy as np
import matplotlib.pyplot as plt

## 1. Setting up the Maze

# Let's create a challenging maze for our hero to navigate!
# Our maze is a 5x5 grid where our hero seeks the exit while avoiding the Minotaur.

MAZE_SIZE = 5
EXIT_LOCATION = (4, 4)  # The exit is in the bottom-right corner
MINOTAUR_LOCATIONS = [(1, 1), (2, 3), (3, 1), (4, 2)]  # Locations where the Minotaur might be

# Define possible actions
ACTIONS = {
    'NORTH': (-1, 0),
    'SOUTH': (1, 0),
    'EAST': (0, 1),
    'WEST': (0, -1)
}

def create_maze():
    """Create and return the initial state of our maze."""
    maze = {}
    for x in range(MAZE_SIZE):
        for y in range(MAZE_SIZE):
            state = (x, y)
            if state == EXIT_LOCATION:
                maze[state] = {'reward': 100, 'type': 'exit'}
            elif state in MINOTAUR_LOCATIONS:
                maze[state] = {'reward': -10, 'type': 'danger'}
            else:
                maze[state] = {'reward': -1, 'type': 'path'}
    return maze

labyrinth = create_maze()

## 2. Visualizing the Maze

def visualize_maze(maze):
    """Create a visual representation of our maze."""
    fig, ax = plt.subplots(figsize=(10, 10))
    for (x, y), info in maze.items():
        if info['type'] == 'exit':
            ax.add_patch(plt.Rectangle((y, MAZE_SIZE-1-x), 1, 1, fill=False, edgecolor='green', lw=3))
            ax.text(y+0.5, MAZE_SIZE-1-x+0.5, 'E', ha='center', va='center', fontsize=20, fontweight='bold', color='green')
        elif info['type'] == 'danger':
            ax.add_patch(plt.Rectangle((y, MAZE_SIZE-1-x), 1, 1, fill=False, edgecolor='red', lw=3))
            ax.text(y+0.5, MAZE_SIZE-1-x+0.5, 'M', ha='center', va='center', fontsize=20, color='red')
    ax.set_xticks(np.arange(0, MAZE_SIZE+1, 1))
    ax.set_yticks(np.arange(0, MAZE_SIZE+1, 1))
    ax.grid(True)
    ax.set_title("Minotaur's Maze")
    plt.show()

visualize_maze(labyrinth)

## 3. Implementing the Markov Decision Process

class MinotaurMazeMDP:
    def __init__(self, maze):
        self.maze = maze
        self.states = list(maze.keys())
        self.actions = list(ACTIONS.keys())

    def get_reward(self, state):
        """Return the reward for the given state."""
        return self.maze[state]['reward']

    def get_transitions(self, state, action):
        """
        Return the possible next states and their probabilities 
        when taking the given action in the given state.
        """
        intended_next_state = self.get_next_state(state, action)
        transitions = []

        # 80% chance of moving in the intended direction
        transitions.append((0.8, intended_next_state))

        # 20% chance (5% each) of moving in other directions or staying put
        for other_action in self.actions:
            if other_action != action:
                other_next_state = self.get_next_state(state, other_action)
                transitions.append((0.05, other_next_state))

        # 5% chance of staying in the same state (maybe the hero hesitates)
        transitions.append((0.05, state))

        return transitions

    def get_next_state(self, state, action):
        """Compute the next state after taking an action."""
        movement = ACTIONS[action]
        next_state = (state[0] + movement[0], state[1] + movement[1])
        
        # Ensure the next state is within the maze boundaries
        next_state = (
            max(0, min(next_state[0], MAZE_SIZE - 1)),
            max(0, min(next_state[1], MAZE_SIZE - 1))
        )
        return next_state

## 4. Value Iteration Algorithm

def value_iteration(mdp, discount_factor=0.9, theta=1e-6):
    """
    Perform value iteration to find the optimal value function.
    
    Args:
    mdp: The MDP object
    discount_factor: How much we value future rewards (default: 0.9)
    theta: The threshold for convergence (default: 1e-6)
    
    Returns:
    A dictionary mapping states to their optimal values
    """
    # Initialize value function
    V = {state: 0 for state in mdp.states}
    
    while True:
        delta = 0
        for state in mdp.states:
            old_v = V[state]
            
            # Compute new value
            action_values = []
            for action in mdp.actions:
                action_value = sum(prob * (mdp.get_reward(next_state) + discount_factor * V[next_state])
                                   for prob, next_state in mdp.get_transitions(state, action))
                action_values.append(action_value)
            
            V[state] = max(action_values)
            
            delta = max(delta, abs(old_v - V[state]))
        
        if delta < theta:
            break
    
    return V

## 5. Extracting the Optimal Policy

def extract_policy(mdp, V, discount_factor=0.9):
    """
    Extract the optimal policy from the optimal value function.
    
    Args:
    mdp: The MDP object
    V: The optimal value function
    discount_factor: How much we value future rewards (default: 0.9)
    
    Returns:
    A dictionary mapping states to their optimal actions
    """
    policy = {}
    for state in mdp.states:
        action_values = []
        for action in mdp.actions:
            action_value = sum(prob * (mdp.get_reward(next_state) + discount_factor * V[next_state])
                               for prob, next_state in mdp.get_transitions(state, action))
            action_values.append((action_value, action))
        
        # Choose the action with the highest value
        best_action = max(action_values)[1]
        policy[state] = best_action
    
    return policy

## 6. Solving and Visualizing the MDP

# Create and solve the MDP
minotaur_maze_mdp = MinotaurMazeMDP(labyrinth)
optimal_values = value_iteration(minotaur_maze_mdp)
optimal_policy = extract_policy(minotaur_maze_mdp, optimal_values)

def visualize_policy(maze, policy):
    """Visualize the optimal policy on the maze."""
    fig, ax = plt.subplots(figsize=(12, 12))
    for (x, y), action in policy.items():
        if maze[(x, y)]['type'] == 'exit':
            ax.add_patch(plt.Rectangle((y, MAZE_SIZE-1-x), 1, 1, fill=False, edgecolor='green', lw=3))
            ax.text(y+0.5, MAZE_SIZE-1-x+0.5, 'E', ha='center', va='center', fontsize=20, fontweight='bold', color='green')
        elif maze[(x, y)]['type'] == 'danger':
            ax.add_patch(plt.Rectangle((y, MAZE_SIZE-1-x), 1, 1, fill=False, edgecolor='red', lw=3))
            ax.text(y+0.5, MAZE_SIZE-1-x+0.5, 'M', ha='center', va='center', fontsize=20, color='red')
        
        if action == 'NORTH':
            ax.arrow(y+0.5, MAZE_SIZE-1-x+0.5, 0, 0.3, head_width=0.3, head_length=0.3, fc='k', ec='k')
        elif action == 'SOUTH':
            ax.arrow(y+0.5, MAZE_SIZE-1-x+0.5, 0, -0.3, head_width=0.3, head_length=0.3, fc='k', ec='k')
        elif action == 'EAST':
            ax.arrow(y+0.5, MAZE_SIZE-1-x+0.5, 0.3, 0, head_width=0.3, head_length=0.3, fc='k', ec='k')
        elif action == 'WEST':
            ax.arrow(y+0.5, MAZE_SIZE-1-x+0.5, -0.3, 0, head_width=0.3, head_length=0.3, fc='k', ec='k')

    ax.set_xticks(np.arange(0, MAZE_SIZE+1, 1))
    ax.set_yticks(np.arange(0, MAZE_SIZE+1, 1))
    ax.grid(True)
    ax.set_title("Optimal Escape Policy for Minotaur's Maze")
    plt.show()

visualize_policy(labyrinth, optimal_policy)

## 7. Analyzing the Results

print("Optimal Values for each state:")
for state, value in optimal_values.items():
    print(f"State {state}: {value:.2f}")

print("\nOptimal Policy:")
for state, action in optimal_policy.items():
    print(f"State {state}: {action}")

## 8. Simulating the Hero's Escape

def simulate_escape(mdp, policy, start_state, max_steps=20):
    """Simulate the hero's escape using the optimal policy."""
    current_state = start_state
    total_reward = 0
    escape_path = [current_state]
    
    for _ in range(max_steps):
        action = policy[current_state]
        next_state = mdp.get_next_state(current_state, action)
        reward = mdp.get_reward(next_state)
        
        total_reward += reward
        escape_path.append(next_state)
        current_state = next_state
        
        if current_state == EXIT_LOCATION:
            break
    
    return escape_path, total_reward

# Simulate an escape
start_state = (0, 0)  # Start from the top-left corner
escape_path, total_reward = simulate_escape(minotaur_maze_mdp, optimal_policy, start_state)

print(f"\nSimulated Escape from {start_state}:")
print(f"Path: {' -> '.join(map(str, escape_path))}")
print(f"Total Reward: {total_reward}")

def visualize_escape(maze, escape_path):
    """Visualize the hero's escape path on the maze."""
    fig, ax = plt.subplots(figsize=(12, 12))
    for (x, y), info in maze.items():
        if info['type'] == 'exit':
            ax.add_patch(plt.Rectangle((y, MAZE_SIZE-1-x), 1, 1, fill=False, edgecolor='green', lw=3))
            ax.text(y+0.5, MAZE_SIZE-1-x+0.5, 'E', ha='center', va='center', fontsize=20, fontweight='bold', color='green')
        elif info['type'] == 'danger':
            ax.add_patch(plt.Rectangle((y, MAZE_SIZE-1-x), 1, 1, fill=False, edgecolor='red', lw=3))
            ax.text(y+0.5, MAZE_SIZE-1-x+0.5, 'M', ha='center', va='center', fontsize=20, color='red')
    
    path_x = [state[1] + 0.5 for state in escape_path]
    path_y = [MAZE_SIZE - 1 - state[0] + 0.5 for state in escape_path]
    ax.plot(path_x, path_y, 'bo-', markersize=10, linewidth=2)
    ax.text(path_x[0], path_y[0], 'Start', ha='right', va='bottom', fontsize=12)
    ax.text(path_x[-1], path_y[-1], 'Exit', ha='left', va='top', fontsize=12)

    ax.set_xticks(np.arange(0, MAZE_SIZE+1, 1))
    ax.set_yticks(np.arange(0, MAZE_SIZE+1, 1))
    ax.grid(True)
    ax.set_title("Hero's Escape Path")
    plt.show()

visualize_escape(labyrinth, escape_path)

## 9. Conclusion and Further Exploration

print("""
Conclusion:
We've successfully modeled and solved the Minotaur's Maze using a Markov Decision Process.
Our hero has learned an optimal policy to navigate the dangerous maze and find the exit.

Further Exploration Ideas:
1. Implement a moving Minotaur that changes position probabilistically.
2. Add multiple exits with different reward values.
3. Introduce "magic items" that give the hero temporary invulnerability.
4. Create a multi-level maze where solving one level leads to another.
5. Implement a Q-learning algorithm for comparison with value iteration.
6. Design a game where a human player competes against the AI hero.

May your hero always find the exit!
""")
