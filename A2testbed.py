# COMP 5801W Assignment 2
# Carleton University
# NOTE: This is a sample script to call your functions and get some results.
#       Change "YourName" to your prefered folder name and put your A2codes.py in the folder.
import time
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

import NicholiShiell.A2codes as A2codes
from A2helpers import RbfFeaturizer, logSoftmaxGradChecker, betaGradChecker, evaluate, render_env, convert1dAction


def _Q2test():

    env_name = "MountainCar-v0"
    env = gym.make(env_name)
    featurizer = RbfFeaturizer(env, 100)

    # Q2b: check the policy visually
    W, eval_returns = A2codes.QLearningFA(env, featurizer, evaluate)
    render_env(env_name, featurizer, W, A2codes.greedyPolicy)

    # Q2c: running experiemnts
    start_time = time.time()
    results = A2codes.runQLExperiments(env,
                                       featurizer,
                                       evaluate)  # this would produce a figure
    print(f'Finished in {time.time() - start_time:.3f} seconds')

    return


def _Q3test():

    env_name = "MountainCar-v0"
    env = gym.make(env_name)
    featurizer = RbfFeaturizer(env, 100)

    # Q3c: check gradient (NOTE: need to set jax.numpy as np in your code when testing this sub-question)
    # s = featurizer.featurize(env.observation_space.sample())
    # a = env.action_space.sample()
    # Theta = np.ones([featurizer.n_features,
    #                  env.action_space.n])  # or any other initialization
    # analytic_grads = A2codes.logSoftmaxPolicyGradient(s, a, Theta)
    # match_grad = logSoftmaxGradChecker(s, a, Theta,
    #                                    A2codes.softmaxProb, analytic_grads)
    # print(f'Gradient matched? {match_grad}')

    # # Q3d: check the policy visually
    Theta, w, eval_returns = A2codes.ActorCritic(env, featurizer, evaluate)
    # # can also try A2codes.greedyPolicy
    render_env(env_name, featurizer, Theta, A2codes.softmaxPolicy)

    # # Q3e: running experiemnts
    # start_time = time.time()
    # results = A2codes.runACExperiments(
    #     env, featurizer, evaluate)  # this would produce a figure
    # print(f'Finished in {time.time() - start_time:.3f} seconds')

    return

def plot_policy(Theta, featurizer):

    ang_points = np.arange(-np.pi, np.pi, 0.1)
    ang_vel_points = np.arange(-8, 8, 0.1)

    # plot the policy
    policy_plot_data = []
    for ang in ang_points:
        for ang_vel in ang_vel_points:
            x = np.cos(ang)
            y = np.sin(ang)
            s = np.array([x, y, ang_vel])
            x_s = featurizer.featurize(s)
            a = A2codes.betaPolicy1d(x_s, Theta, deterministic=True)

            a = 4.*a - 2.

            policy_plot_data.append((ang, ang_vel, a))

    # Create a heatmap of the policy data
    ang_points_grid, ang_vel_points_grid = np.meshgrid(ang_points, ang_vel_points, indexing='ij')
    action_values = np.array([action for _, _, action in policy_plot_data]).reshape(len(ang_points), len(ang_vel_points))

    plt.figure(figsize=(10, 8))
    plt.contourf(ang_points_grid, ang_vel_points_grid, action_values, levels=100, cmap='viridis')
    plt.colorbar(label='Action Value')
    plt.xlabel('Angle (rad)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Policy Heatmap')
    plt.show()

    return ang_points_grid, ang_vel_points_grid, action_values


def plot_path(plot_data, obs_list):

    ang_points_grid, ang_vel_points_grid, action_values = plot_data
    # Extract the angles and angular velocities from the observation list
    angles = [np.arctan2(obs[1],obs[0]) for obs in obs_list]
    ang_velocities = [obs[2] for obs in obs_list]


    plt.figure(figsize=(10, 8))
    plt.contourf(ang_points_grid, ang_vel_points_grid, action_values, levels=100, cmap='viridis')
    plt.colorbar(label='Action Value')
    plt.xlabel('Angle (rad)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Policy Heatmap')

    # Scatter plot of angles and angular velocities
    plt.scatter(angles, ang_velocities, c='red', label='Path', s=10)
    plt.legend()

    plt.show()


def _Q4test():

    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    featurizer = RbfFeaturizer(env, 1000)

    # Q4b: check gradient (NOTE: need to set jax.numpy as np in your code when testing this sub-question)
    # s = featurizer.featurize(env.observation_space.sample())
    # a = 0.5  # or any other number in [0, 1]
    # Theta = np.ones([featurizer.n_features, 2])
    # analytic_grads = A2codes.logBetaPolicy1dGradient(s, a, Theta)
    # match_grad = betaGradChecker(s, a, Theta, analytic_grads)
    # print(f'Gradient matched? {match_grad}')

    # Q4c: check the policy visually
    Theta, w, eval_returns = A2codes.ActorCriticCont(env,
                                                     featurizer,
                                                     evaluate,
                                                     max_episodes=2500,
                                                     evaluate_every=25)

    # Plot the evaluation returns
    plt.plot(eval_returns)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Evaluation Returns Over Episodes')
    plt.grid(True)
    plt.show()

    plot_data = plot_policy(Theta, featurizer)

    def policy_func(x, Theta):
        # can also try deterministic=False
        return convert1dAction(env, A2codes.betaPolicy1d(x, Theta, deterministic=True))

    while True:
        obs_list = render_env(env_name, featurizer, Theta, policy_func)

        plot_path(plot_data, obs_list)

        user_input = input("Do you want to run again? (y/n): ")
        if user_input.lower() == 'n':
            break



    return


if __name__ == "__main__":

    #_Q2test()
    #_Q3test()
    _Q4test()
