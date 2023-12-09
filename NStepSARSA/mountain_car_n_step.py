import numpy as np
import gym
import math
import time
import pickle
import itertools
import matplotlib.pyplot as plt

def epsilonGreedyPolicy(Q, state, epsilon, num_actions):
    # # Find the indices of the optimal actions
    # optimal_actions = np.where(Q[state[0], state[1], :]== np.max(Q[state[0], state[1], :]))[0]
    # num_optimal_actions = len(optimal_actions)

    # # Initialize the action probabilities
    # action_probabilities = np.ones(num_actions) * epsilon / num_actions

    # # Assign higher probability to optimal actions
    # action_probabilities[optimal_actions] += (1 - epsilon) / num_optimal_actions

    # # Choose an action based on the probabilities    
    # chosen_action = np.random.choice(num_actions, p=action_probabilities)

    # return chosen_action

    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[state[0], state[1], :])
    
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
    low = np.array([-1.2, -0.07])
    high = np.array([0.6, 0.07])
    initial_state = np.random.uniform(low=low, high=high)
    
    state = np.array([initial_state[0], initial_state[1]])
    
    return np.array(state)

def nStepSarsa(env, num_episodes, alpha, gamma, epsilon, theta, n, num_bins):
    num_actions = env.action_space.n
    decay_const = 100000
    epsilon_decay = 0.00001
    episode = 0
    delta = 0.0
    seed_value = 50
    M = 4

    # Q = np.random.uniform(low=-1, high=1, size=(num_bins, num_bins, num_actions))
    Q = Q = np.full((num_bins, num_bins, num_actions),0.0)
    steps_per_episode = []
    steps = 0
    steps_list = []
    moving_avg_steps = []
    

    while True:
        episode += 1
        steps2 = 0
        # state = getInitialState()
        # env.reset()
        # env.state = env.unwrapped.state = state
        state = env.reset()
        print(state)
        state = discretizeState(state, num_bins)
        # state = fourierSin(M, state)
        action = epsilonGreedyPolicy(Q, state, epsilon, num_actions)
        episode_states, episode_actions, episode_rewards = [state], [action], [0]

        print(f"Episode {episode}:")
        print(f"Starting state at (discretized): {state}")
        

        T = float('inf')
        for t in itertools.count():
            if t < T:
                next_state, reward, terminated, truncated, info = env.step(action)
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
                    next_action = epsilonGreedyPolicy(Q, next_state, epsilon, num_actions)
                    episode_actions.append(next_action)

            tau = t - n + 1
            if tau >= 0:
                G = sum([episode_rewards[i] * (gamma ** (i - tau - 1)) for i in range(tau + 1, min(tau + n, T) + 1)])
                if tau + n < T:
                    G += (gamma ** n) * Q[episode_states[tau + n][0], episode_states[tau + n][1], episode_actions[tau + n]]

                state_to_update = episode_states[tau][0], episode_states[tau][1], episode_actions[tau]
                change = alpha * (G - Q[state_to_update])
                Q[state_to_update] += change
                delta = max(abs(change),delta)
            if tau == T - 1:
                break

            state = next_state
            action = next_action
        
        alpha *= (1 - 1/(decay_const))
        epsilon *= epsilon_decay
        
        print(f"Max change is {delta}")
        print(f"Steps taken = {steps2}")

        steps_per_episode.append(steps)
        steps_list.append(steps2)
        if episode == num_episodes:
            break
        # time.sleep(1)

        if delta < theta:
            break

        total_return = np.sum(np.array(episode_rewards))
        # print(episode_rewards)
        print(f"Total return is {total_return}")
    
    print(f"Average number of steps taken to reach goal is {np.mean(steps_list)}")
    return Q, steps_list


def runningAverage(steps_list):
    cumulative_sum = 0
    averages = []

    for i, value in enumerate(steps_list, 1):
        cumulative_sum += value
        current_average = cumulative_sum / i
        averages.append(current_average)

    return averages

if __name__ == "__main__":
    max_episode_steps = 5000
    env = gym.make('MountainCar-v0', max_episode_steps=max_episode_steps)
    num_actions = env.action_space.n
    num_episodes = 200
    alpha = 0.01
    gamma = 0.9
    epsilon = 0.9
    theta = 0.0001
    n = 4
    num_bins = 10
    # M = 4
    iterations = 1

    # steps_list = []
    for i in range(iterations):
        print(f"---------Iteration {i}-------------")
        Q, steps_per_episode = nStepSarsa(env, num_episodes, alpha, gamma, epsilon, theta, n, num_bins)
        # steps_list.append(steps_per_episode)

    # Average over iterations
    # steps_per_episode = np.mean(np.array(steps_list),axis=0)

    # Running average on one iteration
    steps_per_episode = runningAverage(steps_per_episode)

    # print(Q)
    # print(steps_per_episode)
    # Plot graphs
    plt.plot(range(1, len(steps_per_episode) + 1), steps_per_episode)
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.title('Episodes vs Steps')
    plt.savefig('nstepsarsa-Train.png')
    plt.show()
    

    print("\n--------Testing---------------")
    # Testing the learned policy
    # env = gym.make('MountainCar-v0', render_mode="human", max_episode_steps=max_episode_steps)
    env = gym.make('MountainCar-v0', render_mode="rgb_array", max_episode_steps=max_episode_steps)

    n_tests = 10
    # test_steps = []

    # for i in range(n_tests):
    state = discretizeState(env.reset(), num_bins)
    # state = fourierSin(M, env.reset())
    total_reward = 0.0
    steps = 0
    while True:
        steps += 1
        action = getBestAction(Q, state)
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

    print(f"Total Reward: {total_reward}")
    print(f"Steps taken: {steps} ")
    # test_steps.append(steps)

    # print(f"It took {np.mean(np.array(test_steps))} steps to reach the goal while testing")
    # Close the environment after running the agent
    env.close()
