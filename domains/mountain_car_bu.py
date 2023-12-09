import numpy as np
import gym
import math
import time
import pickle
import itertools
import matplotlib.pyplot as plt


class Estimator():
    def __init__(self, state_dim, action_dim, learning_rate=0.01):
        self.weights = np.zeros((state_dim, action_dim))
        self.learning_rate = learning_rate

    def predict(self, state):
        # print(state, self.weights)
        # print(np.dot(state, self.weights))
        return np.dot(state, self.weights)

    def update(self, state, action, target):
        prediction = self.predict(state)[action]
        # prediction = np.sum(self.weights[state[0], state[1],])
        error = target - prediction
        # print(self.weights.shape, self.learning_rate * error)
        self.weights[:action] += self.learning_rate * error

# def epsilonGreedyPolicy(estimator, state, epsilon):
    # if np.random.rand() < epsilon:
    #     return np.random.choice(len(estimator.predict(state)))
    # else:
    #     return np.argmax(estimator.predict(state))

def epsilonGreedyPolicy(estimator, epsilon, num_actions):

    def policy_fn(state):
        action_probs = np.ones(num_actions, dtype=float) * epsilon / num_actions
        q_values = estimator.predict(state)
        best_action_idx = np.argmax(q_values)
        action_probs[best_action_idx] += (1.0 - epsilon)
        return action_probs
    return policy_fn
    
def getBestAction(Q, state):
    return np.argmax(Q[state[0], state[1], :])

def normalizeSin(state):
    '''
    Normalize eacg component of the state to the appropriate interval
    '''

    if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], np.ndarray):
        state = state[0]

    position = (state[0] + 0.6) * 2 / 1.8 - 1
    velocity = (state[1] + 0.07) * 2 / (2*0.07) - 1

    return (position, velocity)

def fourierSin(M, state):
    '''
    Logic for Sine state representation
    '''
    normState = normalizeSin(state)
    # M - order of Fourier
    feature = []
    feature.append(1.0)

    for i in range(2):
        for j in range(1,M+1):
            feature.append(math.sin(j*math.pi*normState[i]))

    print("feature",feature)
    return (feature[0],feature[1])

def discretizeState(state, num_bins):
    if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], np.ndarray):
        state = state[0]

    position_bins = np.linspace(-1.2, 0.6, num_bins - 1)
    velocity_bins = np.linspace(-0.07, 0.07, num_bins - 1)
    
    discrete_position = np.digitize(state[0], position_bins)
    discrete_velocity = np.digitize(state[1], velocity_bins)
    
    return (discrete_position, discrete_velocity)

def getInitialState():
    position_range = (-1.2, 0.6)
    velocity_range = (-0.07, 0.07)

    position = np.random.uniform(position_range[0], position_range[1])
    velocity = np.random.uniform(velocity_range[0], velocity_range[1])

    return np.array([position, velocity])

    low = np.array([-1.2, -0.07])
    high = np.array([0.6, 0.07])
    initial_state = np.random.uniform(low=low, high=high)
    
    state = np.array([initial_state[0], initial_state[1]])
    
    return np.array(state)

def getAction(policy, state):
    action_probs = policy(state)
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

    return action

def nStepSarsa(env, num_episodes, alpha, gamma, epsilon, theta, n, num_bins):
    num_actions = env.action_space.n
    decay_const = 100000
    epsilon_decay = 0.00001
    episode = 0
    # delta = 0.0
    # seed_value = 50
    # M = 4

    estimator = Estimator(len(discretizeState(env.reset(), num_bins)), num_actions)
    policy = epsilonGreedyPolicy(estimator, epsilon, env.action_space.n)

    # Q = np.random.uniform(low=-1, high=1, size=(num_bins, num_bins, num_actions))
    Q = Q = np.full((num_bins, num_bins, num_actions),0.0)
    steps_per_episode = []
    steps = 0
    steps_list = []
    

    while True:
        episode += 1
        steps2 = 0
        state = getInitialState()
        env.reset()
        env.state = env.unwrapped.state = state
        # state = env.reset()
        state = discretizeState(state, num_bins)
        action = getAction(policy, state)
        # state = fourierSin(M, state)
        # action = epsilonGreedyPolicy(estimator, state, epsilon)
        episode_states, episode_actions, episode_rewards = [state], [action], [0]

        print(f"Episode {episode}:")
        print(f"Starting state at: {state}")
        

        T = float('inf')
        for t in itertools.count():
            if t < T:
                next_state, reward, terminated, truncated, info = env.step(action)
                # print("next state",next_state)
                next_state = discretizeState(next_state, num_bins)
                # next_state = fourierSin(M, next_state)
                episode_states.append(next_state)
                episode_rewards.append(reward)
                steps += 1
                steps2 += 1

                if terminated:
                    print(f"Reached the goal in {steps2} steps")

                if terminated or truncated:
                    T = t + 1
                else:
                    # next_action = epsilonGreedyPolicy(estimator, state, epsilon)
                    next_action = getAction(policy, next_state)
                    episode_actions.append(next_action)

            tau = t - n + 1
            if tau >= 0:
                G = sum([episode_rewards[i] * (gamma ** (i - tau - 1)) for i in range(tau + 1, min(tau + n, T) + 1)])
                if tau + n < T:
                    q_val = estimator.predict(episode_states[tau + n])
                    G += q_val[episode_actions[tau+n]]

                state_to_update = (episode_states[tau][0], episode_states[tau][1])
                estimator.update(state_to_update, episode_actions[tau], G)

                # change = alpha * (G - Q[state_to_update])
                # Q[state_to_update] += change
                # delta = max(abs(change),delta)
            if tau == T - 1:
                break

            state = next_state
            action = next_action
        
        alpha *= (1 - 1/(decay_const))
        epsilon *= epsilon_decay
        
        # print(f"Max change is {delta}")
        print(f"Steps taken = {steps2}")

        steps_per_episode.append(steps)
        steps_list.append(steps2)
        if episode == num_episodes:
            break
        # time.sleep(1)

        # if delta < theta:
        #     break

        total_return = np.sum(np.array(episode_rewards))
        # print(episode_rewards)
        print(f"Total return is {total_return}")
    
    print(f"Average number of steps taken to reach goal is {np.mean(steps_list)}")
    return estimator, steps_per_episode


if __name__ == "__main__":
    max_episode_steps = 1500
    # env = gym.make('MountainCar-v0', render_mode="human", max_episode_steps=max_episode_steps)
    env = gym.make('MountainCar-v0', max_episode_steps=max_episode_steps)
    # env = gym.make('MountainCar-v0').env
    num_actions = env.action_space.n
    num_episodes = 1000
    alpha = 0.01
    gamma = 1.0
    epsilon = 0.9
    theta = 0.0001
    n = 4
    num_bins = 10
    # M = 4

    estimator, steps_per_episode = nStepSarsa(env, num_episodes, alpha, gamma, epsilon, theta, n, num_bins)
    policy = epsilonGreedyPolicy(estimator, epsilon, env.action_space.n)
    # print(Q)
    # Plot graphs
    plt.plot(steps_per_episode, range(1, len(steps_per_episode) + 1))
    plt.xlabel('Steps')
    plt.ylabel('Episodes')
    plt.title('Episodes vs Steps')
    plt.savefig('nstepsarsa-Train.png')
    plt.show()
    

    # Testing the learned policy
    # env = gym.make('MountainCar-v0', render_mode="human", max_episode_steps=max_episode_steps)
    env = gym.make('MountainCar-v0', render_mode="human").env
    # env = gym.make('MountainCar-v0', max_episode_steps=max_episode_steps)
    state = discretizeState(env.reset(), num_bins)
    # state = fourierSin(M, env.reset())
    total_reward = 0.0
    steps = 0
    while True:
        steps += 1
        action = getAction(policy, state)
        next_state, reward, terminated, truncated, info = env.step(action)
        state = discretizeState(next_state, num_bins)
        # next_state = fourierSin(M, next_state)
        total_reward += reward
        env.render()

        if truncated: 
            print("Was truncated")
            break

        if terminated:
            print("Reached the goal")
            break

    # print(f"Total Reward: {total_reward}")
    # print(f"Steps taken: {steps} ")

    # # Close the environment after running the agent
    env.close()
