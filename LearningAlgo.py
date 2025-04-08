import os
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import digamma,beta
import jax
import matplotlib.pyplot as plt

import gymnasium as gym

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RbfSwarmFeaturizer():
    '''
        This class converts the raw state/obvervation features into
        RBF features. It does a z-score normalization and computes the
        Gaussian kernel values from randomly selected centers.
    '''

    def __init__(self, env, n_features=100):
        
        # we only use the first row of the state
        centers = np.array([env.observation_space.sample()[0]
                            for _ in range(n_features)])
        
        self._mean = np.mean(centers, axis=0, keepdims=True)
        self._std = np.std(centers, axis=0, keepdims=True)
        self._centers = (centers - self._mean) / self._std
        self.n_features = n_features

    def featurize_swarm(self, 
                        state : np.ndarray):
        
        rbf_features = np.zeros((state.shape[0], self.n_features))
        for i, row in enumerate(state):
            rbf_features[i, :] = self.featurize(row)        
        
        return rbf_features
        

    def featurize(self, state):        
        z = state[None, :] - self._mean
        z = z / self._std
                
        dist = cdist(z, self._centers)
        return np.exp(- (dist) ** 2).flatten()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def softplus(x, limit=30):
    """
       Numerically stable softplus function.
       Treat as linear when the input is larger than the limit.
    """
    return jax.numpy.where(x > limit, x, jax.numpy.log1p(jax.numpy.exp(x)))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def softplusGrad(x):
    """
       Gradient of the softplus function, which is sigmoid.
    """
    return jax.nn.sigmoid(x)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def convert1dAction(action_space, a):
    """
       Convert an action a in [0, 1] to an actual action
       specified by the the range of the environment.
       Assumes that the action space is 1d.
    """
    a = (a * (action_space.high[0] - action_space.low[0])
         + action_space.low[0])
    return np.array([a, ])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EPSILON = 1e-6

def betaPolicy1d(   x_s : np.ndarray,
                    theta : np.ndarray,
                    deterministic=False) -> float:

    # calculate the action preference vector for alpha and then beta
    soft_plus_a = softplus(x_s.transpose() @ theta[:,0]) + 1.
    soft_plus_b = softplus(x_s.transpose() @ theta[:,1]) + 1.

    a = 0.
    if deterministic:
        a =  (soft_plus_a-1.) / (soft_plus_a + soft_plus_b + EPSILON)
    else:
        a = np.random.beta(soft_plus_a, soft_plus_b)

    return a

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def logBetaPolicy1dGradient(x_s : np.ndarray,
                            a : float,
                            theta : np.ndarray) -> np.ndarray:

    # calculate the action preference vector for alpha and then beta
    soft_plus_a = softplus(x_s.transpose() @ theta[:,0]) + 1.
    soft_plus_b = softplus(x_s.transpose() @ theta[:,1]) + 1.

    psi_a = digamma(soft_plus_a)
    psi_b = digamma(soft_plus_b)
    psi_a_b = digamma(soft_plus_a + soft_plus_b)

    # calculate the gradient for the alpha parameter
    grad_a = (np.log(a) - psi_a + psi_a_b) * softplusGrad(x_s.transpose() @ theta[:,0]) * x_s

    # calculate the gradient for the beta parameter
    grad_b = (np.log(1. - a) - psi_b + psi_a_b) * softplusGrad(x_s.transpose() @ theta[:,1]) * x_s

    return np.vstack((grad_a, grad_b)).transpose()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_swarm_actions(x_s, theta, action_space, deterministic=False):
    """
       Get the actions for the swarm based on the action preference vector
       and the beta distribution.
       The action is a 1d vector.
    """
    
    action = np.zeros(action_space.shape, dtype=float)

    for i in range(x_s.shape[0]):
        # get the action preference vector        
        action[i, :] = betaPolicy1d(x_s[i, :], theta, deterministic)
    
    return action

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evaluation_run( env,
                    theta, 
                    featurizer):
        
    done = False
    trunc = False
    state, _ = env.reset()
    
    # calculate the state feature vector
    x_s = featurizer.featurize_swarm(state)
    
    largest_blob_list = []
    
    while not (done or trunc):

        action = get_swarm_actions( x_s, 
                                    theta,
                                    env.action_space)

        state, reward, done, trunc, info = env.step(action)

        largest_blob_list.append(np.max(reward))

        # Plot the temperature field and gradient vectors
        env.render()

        # # update the state
        x_s = featurizer.featurize_swarm(state)
        
    return largest_blob_list

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_policy(theta, featurizer, filename):
    ang_points = np.arange(-np.pi, np.pi, 0.1)
    intensity_points = np.arange(0., 100., 0.5)

    # plot the policy
    policy_plot_data = []
    for ang in ang_points:
        for intensity in intensity_points:
            s = np.array([ang, intensity])
            x_s = featurizer.featurize(s)
            # a = convert1dAction(env.action_space, betaPolicy1d(x_s, theta, deterministic=True))
            a = betaPolicy1d(x_s, theta, deterministic=True)
            policy_plot_data.append((ang, intensity, a))
            
    # Create a heatmap of the policy data
    ang_points_grid, ang_vel_points_grid = np.meshgrid(ang_points, intensity_points, indexing='ij')
    action_values = np.array([action for _, _, action in policy_plot_data]).reshape(len(ang_points), len(intensity_points))

    levels = np.linspace(0.,1., 100)
    plt.figure(figsize=(10, 8))
    plt.contourf(ang_points_grid, ang_vel_points_grid, action_values,levels=levels, cmap='viridis')
    plt.colorbar(label='Action Value')
    plt.xlabel('Angle (rad)')
    plt.ylabel('Intensity')
    plt.title('Policy Heatmap')
    plt.savefig(filename)
    plt.close()
    
    return ang_points_grid, ang_vel_points_grid, action_values

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def ActorCriticCont(env,
                    featurizer,
                    max_episodes=1500,
                    evaluate_every=50):

    # policy parameters for a linear function approximation
    # each column is a parameter vector for the beta distribution
    # theta[:,0] is the alpha parameter
    # theta[:,1] is the beta parameter
    theta = np.random.uniform(0., 1., size=(featurizer.n_features, 2))
    #theta = np.ones([featurizer.n_features, 2])

    # state value function weights for a linear function approximation
    w = np.zeros(featurizer.n_features)

    # initialize the average return
    # r_avg = 0.
    r_avg = np.zeros_like(env.reward)

    lambda_w = 0.05
    lambda_theta = 0.05
    alpha_w = 0.001
    alpha_theta = 0.001
    alpha_r = 0.001

    eligibility_w = np.zeros_like(w)
    eligibility_theta = np.zeros_like(theta)

    eval_returns = []
    for episode_idx in range(1, max_episodes + 1):
        # get s_0
        s, _ = env.reset()

        # calculate the state feature vector
        x_s = featurizer.featurize_swarm(s)

        # initialize some parameters
        terminated = truncated = False

        step_counter = 0

        # loop until the episode terminates or is truncated
        s_prime = s
        while not (terminated or truncated):
            print(f"Episode: {episode_idx} / {max_episodes} - Step: {step_counter} / {env.max_steps}")
            step_counter += 1

            # choose an action by sampling the policy distribution parameterized by theta
            # the environment is expecting an action in range [0, 1]
            a = get_swarm_actions(  x_s, 
                                    theta,
                                    env.action_space)

            # take action 'a' and observe the next state and reward
            s_prime, r, terminated, truncated, _ = env.step(a)

            
            # merge a and s_prime into a single array
            merged_action_state = np.hstack((a, s_prime))
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print(merged_action_state)
            
            # feature vector of the next state
            x_s_prime = featurizer.featurize_swarm(s_prime)

            # loop over all agents and apply the actor-critic update
            for agent_idx in range(env.n_agents):

                # if agent_idx != 0:
                #     continue
                # linear value function approximation of s
                v_of_s = w.transpose() @ x_s[agent_idx, :]

                #linear value function approximation of s_prime
                v_of_s_prime = np.zeros_like(v_of_s)
                if not terminated or truncated:
                    v_of_s_prime = w.transpose() @ x_s_prime[agent_idx, :]

                # calculate TD error
                td_error = r[agent_idx] - r_avg[agent_idx] + v_of_s_prime - v_of_s

                # update the average return
                r_avg[agent_idx] = r_avg[agent_idx] + alpha_r * td_error

                # eligibility trace for the actor
                # soft_plus_a = softplus(x_s[agent_idx, :].transpose() @ theta[:,0]) + 1.
                # soft_plus_b = softplus(x_s[agent_idx, :].transpose() @ theta[:,1]) + 1.
                # print(f'{step_counter},{soft_plus_a}, {soft_plus_b}, {td_error}, {r[agent_idx]},{r_avg},{v_of_s},{v_of_s_prime}, {a[agent_idx]}' )
                
                log_beta = logBetaPolicy1dGradient(x_s[agent_idx, :], a[agent_idx], theta)
                
                # eligibility trace for the critic
                eligibility_w = lambda_w * eligibility_w + x_s[agent_idx, :]
                eligibility_theta = lambda_theta * eligibility_theta + log_beta

                # update the critic
                w = w + alpha_w * td_error * eligibility_w

                # update the actor
                theta = theta + alpha_theta *  td_error * eligibility_theta

            # update the current state
            x_s = x_s_prime
            
        # policy_filename = f'policy_episode_{episode_idx:03d}.png'
        # plot_policy(theta, featurizer, filename=policy_filename)

        if episode_idx % evaluate_every == 0:
            evaluation_run(env, theta, featurizer)
            plot_policy(theta, featurizer, filename=f'policy_eval_run_{episode_idx}.png')

    return theta, w, eval_returns

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def ActorCriticEpisodic(env,
                        featurizer,
                        max_episodes=1500,
                        evaluate_every=50):

    # policy parameters for a linear function approximation
    # each column is a parameter vector for the beta distribution
    # theta[:,0] is the alpha parameter
    # theta[:,1] is the beta parameter
    theta = np.random.uniform(0., 1., size=(featurizer.n_features, 2))
  
    # state value function weights for a linear function approximation
    w = np.zeros(featurizer.n_features)

    lambda_w = 0.05
    lambda_theta = 0.05
    alpha_w = 0.001
    alpha_theta = 0.001
    gamma = 0.99

    eval_returns = []
    for episode_idx in range(1, max_episodes + 1):

        eligibility_w = np.zeros_like(w)
        eligibility_theta = np.zeros_like(theta)
        gamma_mx = np.ones_like(theta)

        # get s_0
        s, _ = env.reset()

        # calculate the state feature vector
        x_s = featurizer.featurize_swarm(s)

        # initialize some parameters
        terminated = truncated = False

        # loop until the episode terminates or is truncated
        s_prime = s
        step_counter = 0
        while not (terminated or truncated):
            print(f"Episode: {episode_idx} / {max_episodes} - Step: {step_counter} / {env.max_steps}")
            step_counter += 1

            # choose an action by sampling the policy distribution parameterized by theta
            # the environment is expecting an action in range [0, 1]
            a = get_swarm_actions(  x_s, 
                                    theta,
                                    env.action_space)

            # take action 'a' and observe the next state and reward
            s_prime, r, terminated, truncated, _ = env.step(a)
            
            # feature vector of the next state
            x_s_prime = featurizer.featurize_swarm(s_prime)

            # loop over all agents and apply the actor-critic update
            for agent_idx in range(env.n_agents):

                if agent_idx != 0:
                    continue
                # linear value function approximation of s
                v_of_s = w.transpose() @ x_s[agent_idx, :]

                #linear value function approximation of s_prime
                v_of_s_prime = np.zeros_like(v_of_s)
                if not terminated or truncated:
                    v_of_s_prime = w.transpose() @ x_s_prime[agent_idx, :]

                # calculate TD error
                td_error = r[agent_idx] + gamma*v_of_s_prime - v_of_s
                
                log_beta = logBetaPolicy1dGradient(x_s[agent_idx, :], a[agent_idx], theta)
                
                # eligibility trace for the critic
                eligibility_w = gamma*lambda_w * eligibility_w + x_s[agent_idx, :]
                eligibility_theta = gamma*lambda_theta * eligibility_theta + gamma_mx*log_beta

                # update the critic
                w = w + alpha_w * td_error * eligibility_w

                # update the actor
                theta = theta + alpha_theta *  td_error * eligibility_theta

            gamma_mx = gamma * gamma_mx
            # update the current state
            x_s = x_s_prime
            
        # policy_filename = f'policy_episode_{episode_idx:03d}.png'
        # plot_policy(theta, featurizer, filename=policy_filename)

        if episode_idx % evaluate_every == 0:
            evaluation_run(env, theta, featurizer)
            plot_policy(theta, featurizer, filename=f'policy_eval_run_{episode_idx}.png')
            np.savez(   f"model_eps_{episode_idx}.npz", 
                        Theta=theta, 
                        ft_centers=featurizer._centers, 
                        ft_means=featurizer._mean, 
                        ft_std =featurizer._std) 

    return theta, w, eval_returns
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if __name__ == "__main__":
    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    featurizer = RbfSwarmFeaturizer(env, 100)

    Theta, w, eval_returns = ActorCriticCont(env, featurizer, evaluate)

    def policy_func(x, Theta):
        # can also try deterministic=False
        return convert1dAction(env, betaPolicy1d(x, Theta, deterministic=True))

    render_env(env_name, featurizer, Theta, policy_func)