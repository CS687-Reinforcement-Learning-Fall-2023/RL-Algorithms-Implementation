import gym 
import numpy as np
from heapq import *
from queue import PriorityQueue

class PrioritizedSweepingAgent:
  def __init__(self, env, num_bins, epsilon=0.3, alpha=0.1, n=5, max_episodes=100, theta=0, gamma=1):
        self.env = env
        self.num_bins = num_bins
        self.num_actions = self.env.action_space.n
        self.action_values = np.full((num_bins, num_bins//2, self.num_actions), 0.0)
        # for i in range(num_bins):
        #   for j in range(num_bins//2):
        #      self.action_values[i][j] = np.random.random()

        self.state_actions = []  
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
    r, c = self.state
    
    q = self.action_values[r][c]
    
    optimal_idx = np.where(np.max(q) == q)[0]

    if np.random.uniform(0, 1) <= self.epsilon :
      return np.random.choice(self.num_actions)
    else: 
      return np.random.choice(optimal_idx)

  def reset(self):
      self.state_actions = []
      return self.env.reset()
  
  def s0(self):
     return [np.random.uniform(-1.2, 0.6), 0.0]

  def run_prioritized_sweeping(self):      
    
      for eps in range(self.max_episodes):
        # x = input()
        self.state, _ = self.reset()
        # state = self.env.reset()
        terminated = False
        truncated = False
        while not truncated and not terminated:
            
            self.state = discretizeState(self.state, self.num_bins)
            r, c = self.state
          
            action = self.get_epsilon_greedy_action()  
            self.state_actions.append((self.state, action))
            # print(self.state, action)

            next_state, reward, terminated, truncated, info = self.env.step(action)
            # print("Next state")
            
            next_state = discretizeState(next_state, self.num_bins)
            nr, nc = next_state
            # print("Next state")
            # print(next_state)

            # Insert into queue
            diff = reward + self.gamma * np.max(self.action_values[nr][nc]) - self.action_values[r][c][action]
            if abs(diff) > self.theta:
                self.queue.put((-abs(diff), (self.state, action)))
                # heappush(self.heap, (-abs(diff), (self.state, action)))  # -diff -> (state, action) pop the smallest

            # Update model
            if self.state not in self.model:
                self.model[self.state] = {}
            self.model[self.state][action] = (reward, next_state)
            
            # Update prev
            if next_state not in self.prev:
                self.prev[next_state] = [(self.state, action)]
            else:
                self.prev[next_state].append((self.state, action))
            
            self.state = next_state
            
            # print("Starting inner loop")
            # print(self.heap)
            # print(self.heap[0])
            
            # print(self.model)  
            # print(self.queue.queue)
            # print(seen)
            # x = input()
            
            count = 0
            for _ in range(self.n):
              if self.queue.empty():
              # if len(self.heap) == 0:
                break
              
              _state, _action = self.queue.get()[1]
              # seen.add(_state)
              # _state, _action = heappop(self.heap)[1]
              # if (_state, _action) in seen:
              #    continue
              # seen.add((_state, _action))
              count += 1
              # print(_state,_action)
              _r, _c = _state
              
              _reward, _next_state = self.model[_state][_action]
              _nr, _nc = _next_state
              
              self.action_values[_r][_c][_action] += self.alpha * (_reward + self.gamma * np.max(self.action_values[_nr][_nc]) - self.action_values[_r][_c][_action])
              
              if _state not in self.prev:
                  continue
        
              for (old_state, old_action) in self.prev[_state]:
                  old_reward, _ = self.model[old_state][old_action]
                  old_diff = old_reward + self.gamma * np.max(self.action_values[_state[0]][_state[1]]) - self.action_values[old_state[0]][old_state[1]][old_action]
                  if abs(old_diff) > self.theta:
                      self.queue.put((-abs(old_diff), (old_state, old_action)))
                      # heappush(self.heap, (-abs(old_diff), (old_state, old_action)))
            
          # end of game
          # if eps % 99 == 0:
        # self.epsilon = max(0.1, self.epsilon - 0.005)
        print(self.epsilon)
        print(f"Episode : {eps+1}, Number of actions: {len(self.state_actions)}")
        print_max_action_values(self.action_values)
        # print(self.action_values)
        self.steps_per_episode.append(len(self.state_actions))
        # state = self.reset()
        
        # if eps % 10 == 0:
        #    self.calcOptimalPolicy()
        #    self.render()

  def render(self) -> None:
     state = self.env.reset()
     
     while True:
        state = discretizeState(state, self.num_bins)
        action = self.optimal_policy[state[0]][state[1]]
        next_state, reward, terminated, truncated, info = env.step(action)
        env.render()
        state = next_state
        if truncated or terminated:
           break

     
  def calcOptimalPolicy(self) -> None:
    '''
    Function that sets the optimal policy at 
    the convergence of the value iteration algorithm
    '''
    self.optimal_policy = [[''] * (self.num_bins//2) for _ in range(self.num_bins)]
    for i in range(self.num_bins):
      for j in range(self.num_bins // 2):

        action = np.argmax(self.action_values[i][j])
        self.optimal_policy[i][j] = action

def print_max_action_values(action_values):
  '''Prints the Max Action values'''
  print("Value Function")
  values = []
  for row in action_values:
    value_row = []
    for qs in row:
      value_row.append(np.max(qs))
    values.append(value_row)

  for i in range(len(values)):
    value_strings = [f"{x:.4f}" for x in values[i]]
    row = '\t\t'.join(value_strings)
    print(row)

def discretizeState(state, num_bins):
    if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], np.ndarray):
        state = state[0]

    position_bins = np.linspace(-1.2, 0.6, num_bins-1)
    velocity_bins = np.linspace(-0.07, 0.07, (num_bins // 2)-1)
    
    discrete_position = np.digitize(state[0], position_bins)
    discrete_velocity = np.digitize(state[1], velocity_bins)
    
    return (discrete_position, discrete_velocity)


if __name__ == "__main__":
  alpha = 0.3
  epsilon = 0.2
  n = 10
  max_episodes = 500
  max_episode_steps = 200
  theta = 0.0001
  num_bins = 8
  
  env = gym.make('MountainCar-v0', max_episode_steps=max_episode_steps)
  # env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False)
  
  agent = PrioritizedSweepingAgent(env=env, num_bins=num_bins,epsilon=epsilon, alpha=alpha, n = n, max_episodes=max_episodes, theta=theta, gamma = 1)
  agent.run_prioritized_sweeping()
  # print_max_action_values(final_values)
  
  # state = agent.env.reset()
  # total_reward = 0.0
  # steps = 0
  # while True:
  #     steps += 1
  #     action = 
  #     next_state, reward, terminated, truncated, info = env.step(action)
  #     state = discretizeState(next_state, num_bins)
  #     # next_state = fourierSin(M, next_state)
  #     total_reward += reward
  #     env.render()

  #     if truncated: 
  #         print("Was truncated")
  #         break

  #     if terminated:
  #         print("Reached the goal")
  #         break

  #   print(f"Total Reward: {total_reward}")
  #   print(f"Steps taken: {steps} ")


  


