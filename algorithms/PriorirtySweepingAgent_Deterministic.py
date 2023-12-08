from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
from heapq import *

symbols = {
  'AD': '↓', 
  'AR': '→',
  'AL': '←',
  'AU': '↑',
}

actions = ['AD', 'AR', 'AL', 'AU']
correct = {
  'AD': (1,0), 
  'AR': (0,1),
  'AL': (0,-1),
  'AU': (-1,0),
}

veerLeft = {
  'AD': (0,1),    # AR instead 
  'AR': (-1,0),   # AU instead
  'AL': (1,0),    # AD instead
  'AU': (0,-1),   # AL instead
}

veerRight = {
  'AD': (0,-1),   # AL instead
  'AR': (1,0),    # AD instead
  'AL': (-1,0),   # AU instead
  'AU': (0,1),    # AR instead
}

CORRECT = 0.8
VEER_LEFT = 0.05
VEER_RIGHT = 0.05
BREAK_DOWN = 0.1

DELTA = 0.0001

REWARD = defaultdict(float)
REWARD[(4,4)] = 10.0
REWARD[(4,2)] = -10.0

TERMINAL = [(4,4)]

OBSTACLES = [(2,2), (3,2)]

class Gridworld:
  def __init__(self, gamma=0.9, obstacles=OBSTACLES, reward=REWARD, terminal=TERMINAL) -> None:
    self.state_values = [[0.0] * 5 for _ in range(5)]
    # self.policy = [[random.choice(actions)] * 5 for _ in range(5)]
    # self.optimal_policy = [[''] * 5 for _ in range(5)]
    # self.action_values = [[[0.0]*4] * 5 for _ in range(5)]
    
    self.rows = 5
    self.cols = 5
    self.num_actions = 4
    self.actions = actions
    self.action_index = {'AD':0, 'AR': 1, 'AL':2, 'AU': 3}

    self.action_values = [[] for _ in range(5)]

    # self.action_values = []
    for i in range(self.rows):
      for j in range(self.cols):
        self.action_values[i].append([0.0 for _ in range(4)])
      

  
    self.gamma = gamma
    self.reward = reward 
    self.terminal = terminal
    self.obstacles = obstacles
    #TODO: Remove, not using this
    self.state = self.s0()
    self.is_terminal()

  def s0(self):
    '''
    Randomly initialize the s uniformly in the given gridspace
    '''
    # return (0,0)
    s0 = (random.choice(range(5)), random.choice(range(5)))
  
    while s0 in self.obstacles or s0 in self.terminal:
      s0 = (random.choice(range(5)), random.choice(range(5)))

    return s0
  
  def is_terminal(self):
    self.end = self.state in self.terminal
    return self.end
  
  def get_reward(self):
    return self.reward[self.state]

  def step(self, action: str) -> tuple:
    '''
    Simulates the agent taking an action s in a state s
    
    Inputs:
    -------
    state: The current state (r,c) the agent is in 
    action: The action a taken in the given state 

    Returns:
    --------
    New state (r', c')
    '''

    # prob = random.random()
    # # Move in the correct direction with 0.8 chance
    # if prob < 0.8: 
    #   delta = correct[action]
    # # Veer left to correct direction with 0.05 chance
    # elif prob < 0.85: 
    #   delta = veerLeft[action]
    # # Veer right to correct direction with 0.05 chance
    # elif prob < 0.9:
    #   delta = veerRight[action]
    # # Breaks down and stays in the same place with 0.1 chance
    # else: 
    #   delta = (0,0)
    delta = correct[action]

    r, c = self.state
    dr, dc = delta 
    nr, nc = (r + dr, c + dc)
    
    # Can't run into obstacle
    if (nr,nc) in self.obstacles:
      return self.state
    
    # Can't run into wall
    if nr > 4 or nr < 0:
      return self.state
    
    # Can't run into wall
    if nc > 4 or nc < 0:
      return self.state
    
    self.state = (nr,nc)
    self.end = self.is_terminal()
    return self.state
  
  def reset(self) -> None:
    self.state = self.s0()
    self.end = self.is_terminal()
  
  def get_deterministic_next_state(self, state: tuple, action: str) -> tuple:
    r, c = state
    dr, dc = correct[action] 
    nr, nc = (r + dr, c + dc)
    
    # Can't run into obstacle
    if (nr,nc) in self.obstacles:
      return state
    
    # Can't run into wall
    if nr > 4 or nr < 0:
      return state
    
    # Can't run into wall
    if nc > 4 or nc < 0:
      return state
    
    return (nr,nc)
  
  def is_valid_state(self, state: tuple) -> bool:
    r, c = state
    if r < 0 or r > 4: 
        return False  
    if c < 0 or c > 4: 
        return False 
    if state in self.obstacles:
     return False
      
    return True

  def get_transition_probabilities(self, state: tuple, action: str) -> dict:
    trans_probs = defaultdict(int)
    trans_probs[self.get_deterministic_next_state(state, action)] += 0.8
    trans_probs[self.get_deterministic_next_state(state, veerLeft[action])] += 0.05
    trans_probs[self.get_deterministic_next_state(state, veerRight[action])] += 0.05
    trans_probs[state] += 0.1
    return trans_probs
  

class PrioritizedSweepingAgent:
  def __init__(self, epsilon=0.3, alpha=0.1, n=5, max_episodes=1, theta=0, gamma=0.9):
        self.domain = Gridworld()
        # self.domain.action_values = [[[0.0]*4] * 5 for _ in range(5)]
        self.state_actions = []  # state & action track
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.heap = []
        heapify(self.heap)

        self.n = n
        self.max_episodes = max_episodes  # number of episodes going to play
        self.theta = theta
        self.steps_per_episode = []

        self.model = {}
        self.queue = PriorityQueue()
        self.prev = {}  # nxtState -> list[(curState, Action)...]
  
  def get_epsilon_greedy_action(self) -> str:
    r, c = self.domain.state
    # print(r,c)
    q = self.domain.action_values[r][c]
    
    optimal_idx = np.where(np.max(q) == q)[0]

    if np.random.uniform(0, 1) <= self.epsilon :
      return np.random.choice(self.domain.actions)
    else: 
      return np.random.choice([self.domain.actions[i] for i in optimal_idx])

  def reset(self):
      self.domain.reset()
      self.state_actions = []

  def run_prioritized_sweeping(self):
      for eps in range(self.max_episodes):
        # x = input()
        while not self.domain.end:
            
            state = self.domain.state
            r, c = state
            
            # x = input()
            action = self.get_epsilon_greedy_action()
            
            self.state_actions.append((state, action))

            next_state = self.domain.step(action)
            # if self.domain.end:
            #    break
            reward = self.domain.get_reward()
            nr, nc = next_state

            # Insert into queue
            diff = reward + self.gamma * np.max(self.domain.action_values[nr][nc]) - self.domain.action_values[r][c][self.domain.action_index[action]]
            if abs(diff) > self.theta:
                self.queue.put((-abs(diff), (state, action)))  # -diff -> (state, action) pop the smallest

            # Update model
            if state not in self.model:
                self.model[state] = {}
            self.model[state][action] = (reward, next_state)
            
            # Update prev
            if next_state not in self.prev:
                self.prev[next_state] = [(state, action)]
            else:
                self.prev[next_state].append((state, action))
            
            state = next_state

            # print("Number of updates will be done")
            
            
            # Update q-values n times
            
            for _ in range(self.n):
              
              if self.queue.empty():
              
                break
              # print(self.model)
              # print("queue size:" , self.queue.queue)
              # _delta, (_state, _action) = self.queue.get()
              _state, _action = self.queue.get()[1]
              # _delta, (_state, _action) = heappop(self.heap)
              _r, _c = _state
              # print(_r, _c)
              _reward, _next_state = self.model[_state][_action]
              
              _nr, _nc = _next_state
              # print(self.domain.action_values)
              # print(self.domain.action_values[_r][_c])
              # print("get ready")
              # print(self.domain.action_index[_action])
              # d = input()
              
              self.domain.action_values[_r][_c][self.domain.action_index[_action]] += self.alpha * (_reward + self.gamma * np.max(self.domain.action_values[_nr][_nc]) - self.domain.action_values[_r][_c][self.domain.action_index[_action]])
              # print(self.domain.action_values)
              # c = input()
              # print(_r,_c)

              # loop for all state, action predicted lead to _state
              if _state not in self.prev:
                  continue
        
              for (old_state, old_action) in self.prev[_state]:
                  old_reward, _ = self.model[old_state][old_action]
                  old_diff = old_reward + self.gamma * np.max(self.domain.action_values[_state[0]][_state[1]]) - self.domain.action_values[old_state[0]][old_state[1]][self.domain.action_index[old_action]]
                  if abs(old_diff) > self.theta:
                      self.queue.put((-abs(old_diff), (old_state, old_action)))
            
          # end of game
          # if eps % 99 == 0:
        
        print(f"Episode : {eps+1}, Number of actions: {len(self.state_actions)}")
        print_max_action_values(self.domain.action_values)
        # print(self.domain.action_values)
        self.steps_per_episode.append(len(self.state_actions))
        self.reset()

def print_max_action_values(action_values):
  '''Prints the Max Action values'''
  print("Value Function")
  values = []
  for row in action_values:
    value_row = []
    for qs in row:
      value_row.append(np.max(qs))
    values.append(value_row)

  for i in range(5):
    value_strings = [f"{x:.4f}" for x in values[i]]
    row = '\t\t'.join(value_strings)
    print(row)

if __name__ == "__main__":
  alpha = 0.1
  epsilon = 0.4
  n = 5
  max_episodes = 500
  theta = 0.0
  
  agent = PrioritizedSweepingAgent(alpha=alpha, epsilon=epsilon, max_episodes=max_episodes, n=n, theta=theta)
  agent.run_prioritized_sweeping()
  final_values = agent.domain.action_values
  print_max_action_values(final_values)
  steps_per_eps = agent.steps_per_episode
  to_plot = [steps_per_eps[0]]
  for i in range(1, len(steps_per_eps)):
    to_plot.append(to_plot[-1] + steps_per_eps[i])
  


  plt.figure()
  plt.plot(to_plot, range(1, len(agent.steps_per_episode) + 1))
  plt.ylabel("Episodes")
  plt.xlabel("Action steps")
  plt.show()


  

