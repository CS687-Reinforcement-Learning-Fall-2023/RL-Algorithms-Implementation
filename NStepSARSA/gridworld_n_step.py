from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
from heapq import *
import numpy as np
import itertools
import matplotlib.pyplot as plt

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

left = {
  'AD': 'AR',    # AR instead 
  'AR': 'AU',   # AU instead
  'AL': 'AD',    # AD instead
  'AU': 'AL',   # AL instead
}

right = {
  'AD': 'AL',   # AL instead
  'AR': 'AD',    # AD instead
  'AL': 'AU',   # AU instead
  'AU': 'AR',    # AR instead
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

to_remove = [(2,2), (3,2), (4,4)]

optimal_values = [
    [4.0187, 4.5548, 5.1575, 5.8336, 6.4553],
    [4.3716, 5.0324, 5.8013, 6.6473, 7.3907],
    [3.8672, 4.3900, 0.0000, 7.5769, 8.4637],
    [3.4182, 3.8319, 0.0000, 8.5738, 9.6946],
    [2.9977, 2.9309, 6.0733, 9.6946, 0.0000]
]

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
    return (0,0)
    s0 = (random.choice(range(5)), random.choice(range(5)))
  
    while s0 in self.obstacles or s0 in self.terminal:
      s0 = (random.choice(range(5)), random.choice(range(5)))

    return s0
  
  def is_terminal(self):
    self.end = self.state in self.terminal
    return self.end
  
  def get_reward(self):
    return self.reward[self.state]
  
  def get_reward(self, state):
    return self.reward[state]

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

    prob = random.random()
    # Move in the correct direction with 0.8 chance
    if prob < 0.8: 
      delta = correct[action]
    # Veer left to correct direction with 0.05 chance
    elif prob < 0.85: 
      delta = veerLeft[action]
    # Veer right to correct direction with 0.05 chance
    elif prob < 0.9:
      delta = veerRight[action]
    # Breaks down and stays in the same place with 0.1 chance
    else: 
      delta = (0,0)
    # delta = correct[action]

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
    trans_probs[self.get_deterministic_next_state(state, left[action])] += 0.05
    trans_probs[self.get_deterministic_next_state(state, right[action])] += 0.05
    trans_probs[state] += 0.1
    return trans_probs.items()

def epsilonGreedyPolicy(Q, state, epsilon, num_actions):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[state[0], state[1], :])
    
def getInitialState():
    # return (0,0)
    grid = [(i, j) for i in range(5) for j in range(5)]
    valid_inital_states = [element for element in grid if element not in to_remove]
    return random.choice(valid_inital_states) 

def runningAverage(steps_list):
    cumulative_sum = 0
    averages = []

    for i, value in enumerate(steps_list, 1):
        cumulative_sum += value
        current_average = cumulative_sum / i
        averages.append(current_average)

    return averages

def getValuePolicy(Q):
    values = [ [0.0]*5 for i in range(5)]
    policy = [ [""]*5 for i in range(5)]
    actions = ["AU", "AD", "AR", "AL"]

    for i in range(5):
        for j in range(5):
            values[i][j] = np.max(Q[i, j, :])
            policy[i][j] = actions[np.argmax(Q[i, j, :])]
    
    return values, policy



def displayValues(values):
    '''
    takes the value function and dislays in matrix form
    '''
    for i in range(5):
        for j in range(5):
            print("%.4f\t" % values[i][j], end = "")
        print("\n")

def displayPolicy(policy):
    '''
    takes the policy and displays in matrix form with arrows, goal states marked accordingly
    '''
    for i in range(5):
        for j in range(5):
            action = ""
            if (i,j) in TERMINAL:
                action = "G"
            elif (i,j) in OBSTACLES:
                action = ""
            elif policy[i][j] == "AR":
                action = '\u2192'
            elif policy[i][j] == "AL":
                action = '\u2190'
            elif policy[i][j] == "AU":
                action = '\u2191'
            elif policy[i][j] == "AD":
                action = '\u2193'
            print(action+"\t", end = "")
        print("\n")

def nStepSarsa(env, num_episodes, alpha, gamma, epsilon, theta, n):
    num_actions = 4
    decay_const = 1000
    epsilon_decay = 0.00001
    episode = 0
    delta = 0.0

    Q = np.full((5, 5, num_actions),10.0)
    Q[4, 4] = [0.0]*4  
    Q[2, 2] = [0.0]*4
    Q[3, 2] = [0.0]*4
    steps_per_episode = []
    steps = 0
    steps_list = []
    moving_avg_steps = []
    actions = ["AU", "AD", "AR", "AL"]
    

    while True:
        episode += 1
        steps2 = 0
        env.reset()
        state = getInitialState()
        # print(state)
        action = epsilonGreedyPolicy(Q, state, epsilon, num_actions)
        episode_states, episode_actions, episode_rewards = [state], [action], [0]

        # print(f"Episode {episode}:")
        # print(f"Starting state at (discretized): {state}")
        

        T = float('inf')
        for t in itertools.count():
            if t < T:
                next_state = env.step(actions[action])
                episode_states.append(next_state)
                episode_rewards.append(REWARD[next_state])
                steps += 1
                steps2 += 1

                if next_state in TERMINAL:
                    T = t + 1
                else:
                    next_action = epsilonGreedyPolicy(Q, next_state, epsilon, num_actions)
                    episode_actions.append(next_action)

            tau = t - n + 1
            if tau >= 0:
                G = sum([episode_rewards[i] * (gamma ** (i - tau - 1)) for i in range(tau + 1, min(tau + n, T) + 1)])
                if tau + n < T:
                    G += (gamma ** n) * Q[episode_states[tau + n][0], episode_states[tau + n][1], episode_actions[tau + n]]

                # print(episode_states, episode_actions)
                state_to_update = episode_states[tau][0], episode_states[tau][1], episode_actions[tau]
                change = alpha * (G - Q[state_to_update])
                Q[state_to_update] += change
                delta = max(abs(change),delta)
            if tau == T - 1:
                break

            state = next_state
            action = next_action
        
        # alpha *= (1 - 1/(decay_const))
        epsilon *= epsilon_decay

        # if episode%1000==0:
        #     print("here")
        #     values, _ = getValuePolicy(Q)
        #     displayValues(values)
        #     print(f"Max change is {delta}")
        #     print(f"Steps taken = {steps2}")

        steps_per_episode.append(steps)
        steps_list.append(steps2)
        if episode == num_episodes:
            break
        # time.sleep(1)

        # if delta < theta:
        #     break
    
    print(f"Average number of steps taken to reach goal is {np.mean(steps_list)}")
    return Q, steps_list

if __name__ == "__main__":
    num_episodes = 1000
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.9
    theta = 1.0
    n = 4
    iterations = 20

    env = Gridworld(gamma)
    num_actions = 4
    
    # Q, steps_per_episode = nStepSarsa(env, num_episodes, alpha, gamma, epsilon, theta, n)

    steps_list = []
    for i in range(iterations):
        print(f"---------Iteration {i}-------------")
        Q, steps_per_episode = nStepSarsa(env, num_episodes, alpha, gamma, epsilon, theta, n)
        steps_list.append(steps_per_episode)
        values, policy = getValuePolicy(Q)
        displayPolicy(policy)
        displayValues(values)

    # Average over iterations
    steps_per_episode = np.mean(np.array(steps_list),axis=0)

    values, policy = getValuePolicy(Q)
    
    displayPolicy(policy)
    displayValues(values)
    print("Optimal values")
    displayValues(optimal_values)

    # Average over iterations
    # steps_per_episode = np.mean(np.array(steps_list),axis=0)

    # Running average on one iteration
    # steps_per_episode = runningAverage(steps_per_episode)

    # print(Q)
    # print(steps_per_episode)
    # Plot graphs
    plt.plot(range(1, len(steps_per_episode) + 1), steps_per_episode)
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.title('Episodes vs Steps')
    plt.savefig('gridworld-n-step.png')
    plt.show()