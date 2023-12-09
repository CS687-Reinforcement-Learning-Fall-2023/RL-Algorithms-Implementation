import gym 
import numpy as np
from heapq import *
from queue import PriorityQueue
import matplotlib.pyplot as plt
import copy

class PrioritizedSweepingAgent:
  def __init__(self, env, epsilon=0.3, alpha=0.1, n=5, max_episodes=100, theta=0, gamma=1):
        self.env = env
        self.num_actions = self.env.action_space.n
        self.Q = {}
        for i in range(32):
           for j in range(11):
              for k in range(2):
                 self.Q[(i,j,k)] = [0.0, 0.0]  

        self.state_actions = []  
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.maax_norm = []
        self.tolerance = 0.000001
        
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
    
    
    q = self.Q[self.state]
    optimal_idx = np.where(np.max(q) == q)[0]

    if np.random.uniform(0, 1) < self.epsilon :
      return np.random.choice(self.num_actions)
    else: 
      return np.random.choice(optimal_idx)

  def reset(self):
      self.state_actions = []
      return self.env.reset()

  def run_prioritized_sweeping(self):
  
      for eps in range(self.max_episodes):
        
        
        terminated = False
        truncated = False
        q_old = copy.deepcopy(self.Q)
        # self.env.state = self.env.unwrapped.state = (31,10,1)
        # self.env.state = self.env.unwrapped.state = (np.random.choice(32),np.random.choice(11),np.random.choice(2))
        self.state, _ = self.env.reset()
        while not truncated and not terminated:
            
            
            action = self.get_epsilon_greedy_action()  
            self.state_actions.append((self.state, action))
            
            next_state, reward, terminated, truncated, info = self.env.step(action)
            # nr, nc = next_state
            # print("Next state")
            # print(next_state)

            # Insert into queue
            diff = reward + self.gamma * np.max(self.Q[next_state]) - self.Q[self.state][action]
            if abs(diff) > self.theta:
                self.queue.put((-abs(diff), (self.state, action)))

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
            
            
            count = 0
            

            for _ in range(self.n):
              if self.queue.empty():
              # if len(self.heap) == 0:
                break  
              _state, _action = self.queue.get()[1]
            
              
              
              _reward, _next_state = self.model[_state][_action]
              
              
              self.Q[_state][_action] += self.alpha * (_reward + self.gamma * np.max(self.Q[_next_state]) - self.Q[_state][_action])
            
              
              if _state not in self.prev:
                  continue
        
              for (old_state, old_action) in self.prev[_state]:
                  old_reward, _ = self.model[old_state][old_action]
                  old_diff = old_reward + self.gamma * np.max(self.Q[_state]) - self.Q[old_state][old_action]
                  if abs(old_diff) > self.theta:
                      self.queue.put((-abs(old_diff), (old_state, old_action)))
                      
          # end of game
          # if eps % 99 == 0:
        # self.epsilon = max(0.1, self.epsilon - 0.005)
        # print(self.epsilon)
        print(f"Episode : {eps+1}, Number of actions: {len(self.state_actions)}")
        # print_max_action_values(self.Q)
        max_norm = calculate_max_norm(q_old, self.Q)
        print("Max norm:", max_norm)
        self.steps_per_episode.append(len(self.state_actions))
        state, info = self.reset()
        
        if max_norm < self.tolerance: 
          break
        

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

def calculate_max_norm(q1, q2):
  delta = 0.0
  print(q1)
  print(q2)
  for key in q1:
    # print(q1[key][0] - q2[key][0])
    delta = max(delta, abs(q1[key][0] - q2[key][0]))
    delta = max(delta, abs(q1[key][1] - q2[key][1]))
  return delta      

def print_max_action_values(action_values):
  '''Prints the Max Action values'''
  print("Value Function")
  values = {}
  for key in action_values:
    values[key] = np.max(action_values[key])
  
  for key in values:
    print(f"{key} : {values[key]:.4f}" ) 

def discretizeState(state, num_bins):
    if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], np.ndarray):
        state = state[0]

    position_bins = np.linspace(-1.2, 0.6, num_bins+1)[1:]
    velocity_bins = np.linspace(-0.07, 0.07, (num_bins // 2) + 1)[1:]
    
    discrete_position = np.digitize(state[0], position_bins)
    discrete_velocity = np.digitize(state[1], velocity_bins)
    
    return (discrete_position, discrete_velocity)


if __name__ == "__main__":
  alpha = 0.01
  epsilon = 0.3
  n = 10
  max_episodes = 1500
  max_episode_steps = 200
  theta = 0.001
  # num_bins = 50
  
  # env = gym.make('MountainCar-v0', render_mode="human", max_episode_steps=max_episode_steps)
  env = gym.make('Blackjack-v1', natural=False, sab=True)
  # env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False)
  
  agent = PrioritizedSweepingAgent(env=env,epsilon=epsilon, alpha=alpha, n = n, max_episodes=max_episodes, theta=theta, gamma = 1)
  agent.run_prioritized_sweeping()
  
  plt.figure()
  final_values = agent.Q
  print_max_action_values(final_values)
  steps_per_eps = agent.steps_per_episode
  to_plot = [steps_per_eps[0]]
  for i in range(1, len(steps_per_eps)):
    to_plot.append(to_plot[-1] + steps_per_eps[i])
  plt.plot(to_plot, range(1, len(agent.steps_per_episode) + 1), label=f"epsilon = {epsilon}")
  plt.xlabel("Action steps")
  plt.ylabel("Epsiodes")
  plt.show()

  # plt.plot(range(1,len(agent.mse) + 1), agent.mse, label=f"epsilon = {epsilon}")
