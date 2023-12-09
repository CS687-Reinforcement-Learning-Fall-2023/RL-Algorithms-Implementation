import gym 
import numpy as np
from heapq import *
from queue import PriorityQueue
import matplotlib.pyplot as plt
import copy
from collections import defaultdict

# class ExtendedFrozenLake(gym.envs.toy_text.frozen_lake):
#    def reset(self, initial_state=None):
#       if initial_state:
#         self.state =

class PrioritizedSweepingAgent:
  def __init__(self, env, epsilon=0.3, alpha=0.1, n=5, max_episodes=100, theta=0, gamma=1):
        self.env = env
        self.num_actions = self.env.action_space.n
        # self.Q = {}
        # for i in range(32):
        #    for j in range(11):
        #       for k in range(2):
        #          self.Q[(i,j,k)] = [0.0, 0.0]  
        
        self.Q = [None] * 16
        for i in range(16):
           self.Q[i] = [0.0 for _ in range(self.num_actions)]

        self.state_actions = []  
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.maax_norm = []
        self.tolerance = 0.000001

        self.obstacles = []
        self.terminal = [(1,1), (1,3), (2,3), (3,1), (3,3)]
        self.reward = defaultdict(int)
        self.reward[(3,3)] = 1.0
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
      self.env.reset()

  def run_prioritized_sweeping(self):
      for eps in range(self.max_episodes):
        
        terminated = False
        truncated = False
        # q_old = copy.deepcopy(self.Q)
        # self.env.state = self.env.unwrapped.state = (31,10,1)
        # self.env.state = self.env.unwrapped.state = (np.random.choice(32),np.random.choice(11),np.random.choice(2))
        
        # x = np.random.choice(16)
        # self.env.unwrapped.state = x
        # self.env.state = x
        self.state, _ = self.env.reset()
        while not truncated and not terminated:
            
            action = self.get_epsilon_greedy_action()  
            self.state_actions.append((self.state, action))
            
            next_state, reward, terminated, truncated, info = self.env.step(action)
            

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
            
            # print(self.queue.queue)
            # x = input()

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
        self.epsilon = max(0.1, self.epsilon - 0.005)
        # print(self.epsilon)
        print(f"Episode : {eps+1}, Number of actions: {len(self.state_actions)}")
        # print_max_action_values(self.Q)
        # max_norm = calculate_max_norm(q_old, self.Q)
        # print("Max norm:", max_norm)
        self.steps_per_episode.append(len(self.state_actions))
        self.reset()
        self.env.reset()
        
        # if max_norm < self.tolerance: 
        #   break

def calculate_max_norm(q1, q2):
  delta = 0.0
  print(q1)
  print(q2)
  for key in q1:
    # print(q1[key][0] - q2[key][0])
    delta = max(delta, abs(q1[key][0] - q2[key][0]))
    delta = max(delta, abs(q1[key][1] - q2[key][1]))
  return delta      

def print_optimal_q(values):
   for i in range(4):
    value_strings = [f"{x:.4f}" for x in values[i]]
    row = '\t\t'.join(value_strings)
    print(row)
   
def print_max_action_values(action_values):
  '''Prints the Max Action values'''
  print("Value Function")
  values = []
  for i in range(4):
    row = []
    for j in range(4):
      row.append(np.max(action_values[i*4 + j]))
    values.append(row)
  
  for i in range(4):
    value_strings = [f"{x:.4f}" for x in values[i]]
    row = '\t\t'.join(value_strings)
    print(row)

  return values


if __name__ == "__main__":
  # alpha = 0.2
  alpha_range = [0.1, 0.2, 0.3, 0.5, 0.7]
  epsilon = 0.1
  # epsilon_range = [0.1, 0.3, 0.5, 0.7]
  n = 20
  max_episodes = 100
  max_episode_steps = 200
  theta = 0.0001
  # num_bins = 50
  
  # env = gym.make('MountainCar-v0', render_mode="human", max_episode_steps=max_episode_steps)
  # env = gym.make('Blackjack-v1', natural=False, sab=True)
  env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
  # agent = PrioritizedSweepingAgent(env=env,epsilon=epsilon, alpha=alpha, n = n, max_episodes=max_episodes, theta=theta, gamma = 1)
  # agent.run_prioritized_sweeping()
  # final_values = agent.Q
  # print_max_action_values(final_values)
  
  # plt.figure()

  repeat = 10
  for alpha in alpha_range:
    to_plot_sum = [0.0] * max_episodes
    for i in range(repeat):
      agent = PrioritizedSweepingAgent(env=env,epsilon=epsilon, alpha=alpha, n = n, max_episodes=max_episodes, theta=theta, gamma = 0.99)
      agent.run_prioritized_sweeping()
      final_values = agent.Q
      print_max_action_values(final_values)
      steps_per_eps = agent.steps_per_episode
      to_plot = [steps_per_eps[0]]
      for i in range(1, len(steps_per_eps)):
        to_plot.append(to_plot[-1] + steps_per_eps[i])
      to_plot_sum = np.add(to_plot_sum, to_plot)
    to_plot_sum = np.divide(to_plot_sum,repeat)
    plt.plot(to_plot_sum, range(1, len(agent.steps_per_episode) + 1), label=f"alpha = {alpha}")
    
  plt.xlabel("Action steps")
  plt.ylabel("Epsiodes")
  plt.legend()
  plt.title('Frozen Lake Learning Curve')
  plt.show()

  # plt.figure()

  # repeat = 20
  # for alpha in alpha_range:
  # total_runs = repeat * len(alpha_range)
  # run = 1
  # for alpha in alpha_range:
  # to_plot_sum = [0.0] * max_episodes
  # q_sum = [[0.0]*4 for _ in range(4)]

  # for i in range(repeat):
  #   agent = PrioritizedSweepingAgent(env=env,epsilon=epsilon, alpha=alpha, n = n, max_episodes=max_episodes, theta=theta, gamma = 0.99)
  #   agent.run_prioritized_sweeping()
  #   final_values = agent.Q
  #   q_sum = np.add(q_sum, print_max_action_values(final_values))
  #   steps_per_eps = agent.steps_per_episode
  #   to_plot = [steps_per_eps[0]]
  #   for i in range(1, len(steps_per_eps)):
  #     to_plot.append(to_plot[-1] + steps_per_eps[i])
  #   to_plot_sum = np.add(to_plot_sum, to_plot)
  #   # print(f"Completed runs: {run} / {total_runs}")
  # print("final optimal")
  # print_optimal_q(np.divide(q_sum, repeat))
  # to_plot_sum = np.divide(to_plot_sum,repeat)
  # plt.plot(to_plot_sum, range(1, len(agent.steps_per_episode) + 1), label=f"alpha = {alpha}")
    
  plt.xlabel("Action steps")
  plt.ylabel("Epsiodes")
  plt.legend()
  plt.title('Frozen Lake Learning Curve')
  plt.show()
  # plt.plot(range(1,len(agent.mse) + 1), agent.mse, label=f"epsilon = {epsilon}")
