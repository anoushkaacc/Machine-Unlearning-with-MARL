import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

def simulate_random_agent():
    # 1. Setup Environment
    size = 10
    start = (5, 9) # Middle Top (visual) / Bottom (matrix)? Standardizing on (x,y)
    # Let's use Plot Coordinates: 0..9
    # Start: (5, 9) [Top]
    # Target: (5, 0) [Bottom]
    target = (5, 0)
    
    # Random Obstacles
    obstacles = []
    for _ in range(20):
        ox = random.randint(0, 9)
        oy = random.randint(1, 8) # Avoid start/target row strictly
        if (ox, oy) != start and (ox, oy) != target:
            obstacles.append((ox, oy))
            
    # Simulation Loop
    path_history = []
    state = start
    path_history.append({'pos': state, 'event': 'start', 'step': 0})
    
    max_steps = 100
    for step in range(1, max_steps):
        x, y = state
        tx, ty = target
        
        # Decide Move (Heuristic Random)
        # 1. moves closer to target
        candidates = []
        if ty < y: candidates.append((x, y-1)) # Down
        if ty > y: candidates.append((x, y+1)) # Up
        if tx < x: candidates.append((x-1, y)) # Left
        if tx > x: candidates.append((x+1, y)) # Right
        
        # 2. Random moves (noise)
        all_moves = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
        valid_moves = [m for m in all_moves if 0 <= m[0] < size and 0 <= m[1] < size]
        
        # Policy: 70% pick closer, 30% random
        if random.random() < 0.7 and candidates:
            # Pick a candidate that is valid
            valid_candidates = [c for c in candidates if 0 <= c[0] < size and 0 <= c[1] < size]
            if valid_candidates:
                next_pos = random.choice(valid_candidates)
            else:
                next_pos = random.choice(valid_moves)
        else:
            next_pos = random.choice(valid_moves)
            
        # Check Event
        if next_pos in obstacles:
            # COLLISION
            path_history.append({'pos': next_pos, 'event': 'boom', 'step': step})
            # Reset
            state = start
            path_history.append({'pos': state, 'event': 'reset', 'step': step})
        elif next_pos == target:
            # GOAL
            path_history.append({'pos': next_pos, 'event': 'goal', 'step': step})
            break
        else:
            # MOVE
            state = next_pos
            path_history.append({'pos': state, 'event': 'move', 'step': step})
            
    return path_history, obstacles, target

# Generate Data
history, obstacles, target = simulate_random_agent()

# Animate
fig, ax = plt.subplots(figsize=(6,6))

def update(frame):
    ax.clear()
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_xticks(np.arange(-0.5, 10, 1))
    ax.set_yticks(np.arange(-0.5, 10, 1))
    ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Static
    ax.add_patch(plt.Rectangle((target[0]-0.5, target[1]-0.5), 1, 1, color='green', alpha=0.3))
    ax.text(target[0], target[1], 'T', color='green', weight='bold', ha='center', va='center')
    
    for ox, oy in obstacles:
        ax.add_patch(plt.Rectangle((ox-0.5, oy-0.5), 1, 1, color='black'))
        
    # Dynamic
    if frame < len(history):
        entry = history[frame]
        pos = entry['pos']
        evt = entry['event']
        
        if evt == 'boom':
            ax.add_patch(plt.Circle(pos, 0.4, color='red'))
            ax.text(pos[0], pos[1], 'BOOM!', color='yellow', weight='bold', ha='center', va='center', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
            ax.set_title(f"Step {entry['step']}: CRASH! Resetting...")
        elif evt == 'goal':
            ax.add_patch(plt.Circle(pos, 0.4, color='gold'))
            ax.set_title("TARGET REACHED!")
        elif evt == 'reset':
             ax.add_patch(plt.Circle(pos, 0.3, color='blue', alpha=0.5))
        else: # move/start
            ax.add_patch(plt.Circle(pos, 0.3, color='blue'))
            ax.set_title(f"Step {entry['step']}: Navigating...")

# Save
total_frames = len(history) + 10
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=400)
ani.save('agent_demo.gif', writer='pillow')
print("Simulation saved to agent_demo.gif")
