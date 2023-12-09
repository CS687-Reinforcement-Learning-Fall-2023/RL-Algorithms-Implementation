import numpy as np
import gym
import random
import math
import time
import pickle
import itertools
import matplotlib.pyplot as plt

def epsilonGreedyPolicy(Q, state, epsilon, num_actions):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[state, :])
    
def getBestAction(Q, state):
    return np.argmax(Q[state, :])

def nStepSarsa(env, num_episodes, alpha, gamma, epsilon, theta, n):
    num_actions = env.action_space.n
    decay_const = 1000
    epsilon_decay = 0.005
    episode = 0
    delta = 0.0
    M = 4
    Q = np.full((16,4),0.0)
    steps_per_episode = []
    steps = 0
    steps_list = []
    moving_avg_steps = []
    

    while True:
        episode += 1
        steps2 = 0
        state = env.reset()
        state = state[0]
        # state = 15
        # env.state = env.unwrapped.state = state
        print(state)
        action = epsilonGreedyPolicy(Q, state, epsilon, num_actions)
        episode_states, episode_actions, episode_rewards = [state], [action], [0]

        if episode%100:
            print(f"Episode {episode}:")
        # print(f"Starting state at (discretized): {state}")
        

        T = float('inf')
        for t in itertools.count():
            if t < T:
                next_state, reward, terminated, truncated, info = env.step(action)
                print("next", next_state)
                episode_states.append(next_state)
                episode_rewards.append(reward)
                steps += 1
                steps2 += 1

                if terminated and state == 15:
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
                    G += (gamma ** n) * Q[episode_states[tau + n], episode_actions[tau + n]]

                state_to_update = episode_states[tau], episode_actions[tau]
                change = alpha * (G - Q[state_to_update])
                Q[state_to_update] += change
                delta = max(abs(change),delta)
            if tau == T - 1:
                break

            state = next_state
            action = next_action
        
        # alpha *= (1 - 1/(decay_const))
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
    max_episode_steps = 100
    env = gym.make('FrozenLake-v1', max_episode_steps=max_episode_steps, is_slippery=False)
    num_actions = env.action_space.n
    num_episodes = 15000
    alpha = 0.8
    gamma = 0.95
    epsilon = 1.0
    theta = 0.0001
    n = 8
    # M = 4
    iterations = 1

    # steps_list = []
    for i in range(iterations):
        print(f"---------Iteration {i}-------------")
        Q, steps_per_episode = nStepSarsa(env, num_episodes, alpha, gamma, epsilon, theta, n)
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
    plt.savefig('frozen-lake.png')
    plt.show()
    

    print("\n--------Testing---------------")
    # Testing the learned policy
    env = gym.make('FrozenLake-v1', render_mode="human", max_episode_steps=max_episode_steps, is_slippery=False)
    # env = gym.make('FrozenLake-v1', render_mode="rgb_array", max_episode_steps=max_episode_steps)

    n_tests = 10
    # test_steps = []

    # for i in range(n_tests):
    # state = discretizeState(env.reset(), num_bins)
    state = env.reset()
    state = state[0]
    # print(state)
    # state = fourierSin(M, env.reset())
    total_reward = 0.0
    steps = 0
    while True:
        steps += 1
        action = getBestAction(Q, state)
        next_state, reward, terminated, truncated, info = env.step(action)
        print(next_state)
        # state = discretizeState(next_state, num_bins)
        # next_state = fourierSin(M, next_state)
        total_reward += reward
        env.render()

        if truncated: 
            print("Was truncated")
            break

        if terminated:
            if state == 15:
                print("Reached the goal")
            break

    print(f"Total Reward: {total_reward}")
    print(f"Steps taken: {steps} ")
    # test_steps.append(steps)

    # print(f"It took {np.mean(np.array(test_steps))} steps to reach the goal while testing")
    # Close the environment after running the agent
    env.close()
