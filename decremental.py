import sys
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
from collections import deque
import random
import json
import torch.nn.functional as F
import pickle
from tqdm import tqdm

# seed = 3
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)

unlearn_epoch = int(sys.argv[1])
n_maps = 20
game_type = "grid_world"

class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)



class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(input_dim, 128)
        self.lin2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.lin1(x))
        x = self.lin2(x)
        return x




class Simple2DEnvironment(gym.Env):
    def __init__(self, size=10, n_ob=10):
        super(Simple2DEnvironment, self).__init__()

        self.size = size
        self.n_ob = n_ob
        # define state space
        self.action_space = spaces.Discrete(4)
        # observation space
        self.observation_space = spaces.Box(low=-1, high=self.size, shape=(10,), dtype=np.int32)
        # define state 
        self.state = None
        # define obstacles
        self.obstacles = None
        # obstacles target
        self.target = None

        self.maps = []
        
        
    def save_maps(self, filename):
        with open(filename, 'w') as f:
            maps = [[list(obstacle) for obstacle in map_data["obstacles"]] + [list(map_data["target"])] for map_data in self.maps]
            json.dump(maps, f)

    def load_maps(self, filename):
        with open(filename, 'r') as f:
            maps = json.load(f)
            self.maps = [{"obstacles": [tuple(obstacle) for obstacle in map_data[:-1]], "target": tuple(map_data[-1])} for map_data in maps]


    def step(self, action):
        x, y = self.state
        old_state = self.state

        # update state
        if action == 0:  # up
            new_y = min(y+1, self.size-1)
            y = new_y
        elif action == 1:  # down
            new_y = max(y-1, 0)
            y = new_y
        elif action == 2:  # left
            new_x = max(x-1, 0)
            x = new_x
        elif action == 3:  # right
            new_x = min(x+1, self.size-1)
            x = new_x

        self.state = (x, y)

        done = self.state == self.target  # done if arrive target
        if self.state == self.target or self.state[1] == 0:
            arrive_target = True
            arrive_land = True
            done = True
        else:
            arrive_target = False
            arrive_land = False
            done = False
            

        reward = -10 if self.state in self.obstacles else 100 if arrive_target else -100 if arrive_land else -5
        # If hit an obstacle, the state returns to the state before the collision
        if self.state in self.obstacles:
            self.state = old_state

        state = self.get_surrounding_cells()
        state = np.insert(state, 0, self.state)
        return state, reward, done, {}

    def reset(self, map_index=None,game_type = "grid_world"):
        if map_index is None:
            # create or load your own map and environment
            if game_type == "grid_world":
                self.state = (int(self.size / 2), self.size - 1)  # Middle Bottom
                self.target = (int(self.size / 2), 0)  # Middle Top

                self.obstacles = [(np.random.randint(1, self.size-1), np.random.randint(1, self.size-1)) for _ in range(self.n_ob)]
                self.obstacles += [(i, -1) for i in range(self.size+2)] # Top
                self.obstacles += [(i, self.size) for i in range(self.size+2)] # Bottom
                self.obstacles += [(-1, i) for i in range(-1, self.size+1)] # Left
                self.obstacles += [(self.size, i) for i in range(-1, self.size+1)] # Right

                map_data = {"obstacles": self.obstacles, "target": self.target}
                self.maps.append(map_data)
            elif game_type == "aircraft_landing":
                self.state = (np.random.randint(self.size), self.size - 1)
                self.target = (np.random.randint(self.size), 0)

                self.obstacles = [(np.random.randint(self.size), np.random.randint(self.size)) for _ in range(10)]
                self.obstacles += [(i, -1) for i in range(self.size+2)] # Top
                self.obstacles += [(i, self.size) for i in range(self.size+2)] # Bottom
                self.obstacles += [(-1, i) for i in range(-1, self.size+1)] # Left
                self.obstacles += [(self.size, i) for i in range(-1, self.size+1)] # Right

                map_data = {"obstacles": self.obstacles, "target": self.target}
                self.maps.append(map_data)
        else:
            # if map index is provided, load it
            map_data = self.maps[map_index]
            self.state = (int(self.size/2), self.size-1) if game_type == "grid_world" else (np.random.randint(self.size), self.size - 1)
            self.target = map_data['target']
            self.obstacles = map_data['obstacles']

        state = self.get_surrounding_cells()
        state = np.insert(state, 0, self.state)
        return state

    
      
    
    def get_surrounding_cells(self):
        surrounding = np.full(8, 0)  # initialize with -1
        x, y = self.state
        directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)] 
        for i, (dx, dy) in enumerate(directions):
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                if (new_x, new_y) == self.state:
                    surrounding[i] = 1  # agent
                elif (new_x, new_y) == self.target:
                    surrounding[i] = 2  # target
                elif (new_x, new_y) in self.obstacles:
                    surrounding[i] = 3  # obstacle
            else:
                surrounding[i] = 3  # edge is considered as obstacle
        return surrounding




    def render(self, mode='human'):
        for y in range(self.size+1, -2, -1):  
            for x in range(-1, self.size+1):  
                if self.state == (x, y):
                    print('X', end='')
                elif self.target == (x, y):
                    print('T', end='')
                elif (x, y) in self.obstacles:
                    print('#', end='')
                else:
                    print(' ', end='')
            print()

class DQNAgent():
    def __init__(self, state_dim, action_dim, replay_buffer):
        self.action_dim = action_dim
        self.gamma = 0.99
        self.model = DQN(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.replay_buffer = replay_buffer
        self.loss_fn = nn.MSELoss(reduction='mean')

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.LongTensor(action).to(device)
        reward     = torch.FloatTensor(reward).to(device)
        done       = torch.FloatTensor(done).to(device)

        q_values      = self.model(state)
        next_q_values = self.model(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = self.loss_fn(q_value, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(self.action_dim), epsilon*(0.75/0.25)
        state   = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_value = self.model.forward(state)
        probabilities = F.softmax(q_value).squeeze(0).tolist()
        action = q_value.max(1)[1].data[0].item()
        return q_value.max(1)[1].data[0], (1-epsilon)*((1-probabilities[action])/probabilities[action])

'''
Normal Model
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = Simple2DEnvironment(10,10)
'''
Load Map
'''
env.load_maps(f"map_data/{game_type}/maps.json")

state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n
buffer = ReplayBuffer(1000)
agent = DQNAgent(state_dim, action_dim, buffer)

episodes = 1000
batch_size = 32
epsilon = 1.0
rewards = []    

'''
Training Normal Model
'''
print("Training Normal Model")
map_id = -1
all_step = 0
steps = 0 
n_episodes = 1000
n_maps = 20
each_map = n_episodes/n_maps
total_reward = 0
for i_episode in tqdm(range(n_episodes)):
    all_step = all_step + steps
    if i_episode%each_map ==0 :
        map_id = map_id + 1
        state = env.reset(map_index=map_id)

        all_step = 0
        epsilon = 1.0
        total_reward = 0
    else:
        state = env.reset(map_index=map_id)
    done = False
    steps = 0  
    while not done and steps < 200:
        action,_ = agent.get_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        agent.update(batch_size)
        total_reward += reward
        state = next_state
        steps += 1

    if (i_episode+1)%each_map ==0 :
        print(f'Map {map_id} Reward: {total_reward/each_map} steps: {all_step/each_map}')
    rewards.append(total_reward)
    epsilon *= 0.995

'''
Test Normal Model
'''
print("Testing Normal Model")
r_truth_normal = []
previous_r_truth_normal_sum = 0
map_id = -1
all_step = 0
# test
steps = 0 
n_episodes = 1000
n_maps = 20
each_map = n_episodes/n_maps
total_reward = 0
epsilon = 0
rewards_normal = []
for i_episode in tqdm(range(n_episodes)):
    all_step = all_step + steps
    if i_episode%each_map ==0:
        map_id = map_id + 1
        state = env.reset(map_index=map_id)

        all_step = 0
        epsilon = 1.0
        total_reward = 0
    else:
        state = env.reset(map_index=map_id)
        
    done = False
    steps = 0 
    previous_r_truth_normal_sum = 0
    while not done and steps < 100:

        action,r_truth = agent.get_action(state, epsilon=0.05)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        previous_r_truth_normal_sum += r_truth
        state = next_state
        steps += 1
    rewards_normal.append(total_reward)
    r_truth_normal.append(previous_r_truth_normal_sum/3)
    if (i_episode+1)%each_map ==0 :
        print(f'Map {map_id} Reward: {total_reward/each_map} steps: {all_step/each_map}')


def shuffle_three_parts(lst,change):
    indices = [i for i, x in enumerate(lst[2:], start=2) if x >-5]

    chosen_indices = np.random.choice(indices, change, replace=False)

    np.random.shuffle(chosen_indices)

    lst_new = lst.copy()
    for i, index in enumerate(sorted(chosen_indices)):
        lst_new[index] = lst[chosen_indices[i]]

    return lst_new

'''
Unlearning Model
'''
print("Training Unlearning Model")
env.maps.clear()
map_id = -1
all_step = 0
# unlearning
steps = 0
n_episodes = unlearn_epoch * n_maps # epoch = 10, should change to 200 400 600 800 1000
n_maps = 20
each_map = n_episodes/n_maps
total_reward = 0
for i_episode in tqdm(range(n_episodes)):
    all_step = all_step + steps
    if i_episode%each_map ==0 :
        map_id = map_id + 1
        state = env.reset() if game_type == "grid_world" else env.reset(None,"aircraft_landing")

        all_step = 0
        epsilon = 1.0
        total_reward = 0
    else:

        state = env.reset(map_index=map_id)
    done = False
    steps = 0
    while not done and steps < 100:

        if map_id == 0:
            action,_ = agent.get_action(state, 1)
            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            agent.update(batch_size)
        else:
            action,_ = agent.get_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)

            agent.update(batch_size)
        total_reward += reward
        state = next_state
        steps += 1

    if (i_episode+1)%each_map ==0 :
        print(f'Map {map_id} Reward: {total_reward/each_map} steps: {all_step/each_map}')
    rewards.append(total_reward)
    epsilon *= 0.995


'''
Unlearning Testing
'''
print("Testing Unlearning Model")
env.load_maps(f"map_data/{game_type}/maps.json")
r_truth_unlearn = []
previous_r_truth_unlearn_sum = 0
map_id = -1
all_step = 0

steps = 0 
n_episodes = 1000
n_maps = 20
each_map = n_episodes/n_maps
total_reward = 0
epsilon = 0
rewards_unlearning = []
# print('test')
for i_episode in tqdm(range(n_episodes)):
    all_step = all_step + steps
    if i_episode%each_map ==0 :
        map_id = map_id + 1
        state = env.reset(map_index=map_id)

        all_step = 0
        epsilon = 1.0
        total_reward = 0
    else:
        state = env.reset(map_index=map_id)

    done = False
    steps = 0
    previous_r_truth_unlearn_sum = 0
    while not done and steps < 100:
        action,r_truth = agent.get_action(state, epsilon=0.05)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        previous_r_truth_unlearn_sum+=r_truth
        state = next_state
        steps += 1
    rewards_unlearning.append(total_reward)
    r_truth_unlearn.append(previous_r_truth_unlearn_sum/3)

    if (i_episode+1)%each_map ==0 :
        print(f'Map {map_id} Reward: {total_reward/each_map} steps: {all_step/each_map}')


'''
Note: During unlearn model training process, we will comment the code to save time 
Different training unlearned model epoch will have the same retain result,therefore we can comment the code below
'''

# '''
# Retain Model
# '''
# print("Training Retain Model")
# env = Simple2DEnvironment(10, 10)
# env.load_maps(f"map_data/{game_type}/maps.json")
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
# buffer = ReplayBuffer(1000)
# agent = DQNAgent(state_dim, action_dim, buffer)
#
# episodes = 1000
# batch_size = 32
# epsilon = 1.0
# rewards = []
#
# '''
# Retain Training - exclude first env
# '''
# map_id = -1
# all_step = 0
# steps = 0
# n_episodes = 1000
# n_maps = 20
# each_map = n_episodes / n_maps
# total_reward = 0
# for i_episode in tqdm(range(n_episodes)):
#     all_step = all_step + steps
#     if i_episode % each_map == 0:
#
#         map_id = map_id + 1
#         state = env.reset(map_index=map_id)
#
#         all_step = 0
#         epsilon = 1.0
#         total_reward = 0
#     else:
#
#         state = env.reset(map_index=map_id)
#     done = False
#     steps = 0
#     while not done and steps < 200:
#         action,_ = agent.get_action(state, epsilon)
#         next_state, reward, done, _ = env.step(action)
#         if map_id != 0:
#             buffer.push(state, action, reward, next_state, done)
#             agent.update(batch_size)
#         total_reward += reward
#         state = next_state
#         steps += 1
#
#     if (i_episode + 1) % each_map == 0:
#         print(f'Map {map_id} Reward: {total_reward / each_map} steps: {all_step / each_map}')
#     rewards.append(total_reward)
#     epsilon *= 0.995
#
# '''
# Retain Test
# '''
# print("Testing Retain Model")
# r_truth_retain = []
# previous_r_truth_retain_sum = 0
# map_id = -1
# all_step = 0
# steps = 0
# n_episodes = 1000
# n_maps = 20
# each_map = n_episodes / n_maps
# total_reward = 0
# epsilon = 0
# rewards_retain = []
# for i_episode in tqdm(range(n_episodes)):
#     all_step = all_step + steps
#     if i_episode % each_map == 0:
#         map_id = map_id + 1
#         state = env.reset(map_index=map_id)
#
#         all_step = 0
#         epsilon = 1.0
#         total_reward = 0
#         # env.render()
#     else:
#         state = env.reset(map_index=map_id)
#
#     done = False
#     steps = 0
#     previous_r_truth_retain_sum = 0
#     while not done and steps < 100:
#         action,r_truth = agent.get_action(state, epsilon=0.05)
#         next_state, reward, done, _ = env.step(action)
#         total_reward += reward
#         # epoch_reward+=reward
#         previous_r_truth_retain_sum += r_truth
#         state = next_state
#         steps += 1
#     rewards_retain.append(total_reward)
#     r_truth_retain.append(previous_r_truth_retain_sum/3)
#     if (i_episode + 1) % each_map == 0:
#         print(f'Map {map_id} Reward: {total_reward / each_map} steps: {all_step / each_map}')

# Note: s3 refers to seed 3, s3 used to be the best seed but now we are not using the seed
'''
dump r_truth
'''
# pickle.dump(r_truth_normal,open(f"s3-dec-RTruth-Normal.pkl","wb"))
pickle.dump(r_truth_unlearn,open(f"s3-dec-RTruth-Unlearn-epoch-{unlearn_epoch}.pkl","wb"))
# pickle.dump(r_truth_retain,open(f"s3-dec-RTruth-Retain.pkl","wb"))

'''
dump reward
'''
# pickle.dump(rewards_normal,open(f"s3-dec-cRewards-Normal.pkl","wb"))
pickle.dump(rewards_unlearning,open(f"s3-dec-cRewards-Unlearn-epoch-{unlearn_epoch}.pkl","wb"))
# pickle.dump(rewards_retain, open(f"s3-dec-cRewards-Retain.pkl", "wb"))
