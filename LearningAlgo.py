import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import digamma
import jax

import gymnasium as gym

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RbfFeaturizer():
    '''
        This class converts the raw state/obvervation features into
        RBF features. It does a z-score normalization and computes the
        Gaussian kernel values from randomly selected centers.
    '''

    def __init__(self, env, n_features=100):
        centers = np.array([env.observation_space.sample()
                            for _ in range(n_features)])
        self._mean = np.mean(centers, axis=0, keepdims=True)
        self._std = np.std(centers, axis=0, keepdims=True)
        self._centers = (centers - self._mean) / self._std
        self.n_features = n_features

    def featurize(self, state):
        z = state[None, :] - self._mean
        z = z / self._std
        dist = cdist(z, self._centers)
        return np.exp(- (dist) ** 2).flatten()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evaluate(env, featurizer, W, policy_func, n_runs=10):
    '''
        Evaluate the policy given the parameters W and policy function.
        Run the environment several times and collect the return.
    '''
    all_returns = np.zeros([n_runs])
    for i in range(n_runs):
        observation, info = env.reset()
        return_to_go = 0
        while True:
            # Agent
            observation = featurizer.featurize(observation)
            action = policy_func(observation, W)

            observation, reward, terminated, truncated, info = env.step(action)
            return_to_go += reward
            if terminated or truncated:
                break
        all_returns[i] = return_to_go

    return np.mean(all_returns)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def render_env(env_name, featurizer, W, policy_func):

    env = gym.make(env_name, render_mode="human")
    observation, info = env.reset()
    while True:
        env.render()
        observation = featurizer.featurize(observation)
        action = policy_func(observation, W)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()
    return

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

def convert1dAction(env, a):
    """
       Convert an action a in [0, 1] to an actual action
       specified by the the range of the environment.
       Assumes that the action space is 1d.
    """
    a = (a * (env.action_space.high[0] - env.action_space.low[0])
         + env.action_space.low[0])
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

def ActorCriticCont(env,
                    featurizer,
                    eval_func,
                    actor_step_size=0.005,
                    critic_step_size=0.005,
                    max_episodes=1500,
                    evaluate_every=50):

    # policy parameters for a linear function approximation
    theta = np.ones([featurizer.n_features, 2])

    # state value function weights for a linear function approximation
    w = np.zeros(featurizer.n_features)

    # initialize the average return
    r_avg = 0.

    lambda_w = 0.001
    lambda_theta = 0.001
    alpha_w = actor_step_size
    alpha_theta = critic_step_size
    alpha_r = 0.005

    eligibility_w = np.zeros_like(w)
    eligibility_theta = np.zeros_like(theta)

    eval_returns = []
    for i in range(1, max_episodes + 1):
        # get s_0
        s, _ = env.reset()

        # calculate the state feature vector
        x_s = featurizer.featurize(s)

        # initialize some parameters
        terminated = truncated = False

        # loop until the episode terminates or is truncated
        while not (terminated or truncated):

            # choose an action by sampling the policy distribution parameterized by theta
            a = betaPolicy1d(x_s, theta)

            # take action 'a' and observe the next state and reward
            s_prime, r, terminated, truncated, _ = env.step(convert1dAction(env, a))

            # feature vector of the next state
            x_s_prime = featurizer.featurize(s_prime)

            # linear value function approximation of s
            v_of_s = w.transpose() @ x_s

            #linear value function approximation of s_prime
            # v_of_s_prime = w.transpose() @ x_s_prime
            v_of_s_prime = np.zeros_like(v_of_s)
            if not terminated or truncated:
                v_of_s_prime = w.transpose() @ x_s_prime

            # calculate TD error
            td_error = r - r_avg + v_of_s_prime - v_of_s

            # update the average return
            r_avg = r_avg + alpha_r * td_error

            # eligibility trace for the critic
            eligibility_w = lambda_w * eligibility_w + x_s

            # eligibility trace for the actor
            eligibility_theta = lambda_theta * eligibility_theta + logBetaPolicy1dGradient(x_s, a, theta)

            # update the critic
            w = w + alpha_w * td_error * eligibility_w

            # update the actor
            theta = theta + alpha_theta *  td_error * eligibility_theta

            # # update the current state
            x_s = x_s_prime

        if i % evaluate_every == 0:
            def policy_func(x, Theta):
                # can also try deterministic=False
                return convert1dAction(env, betaPolicy1d(x, Theta, deterministic=True))

            eval_return = eval_func(env, featurizer, theta, policy_func)
            eval_returns.append(eval_return)

    return theta, w, eval_returns

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if __name__ == "__main__":
    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    featurizer = RbfFeaturizer(env, 100)

    Theta, w, eval_returns = ActorCriticCont(env, featurizer, evaluate)

    def policy_func(x, Theta):
        # can also try deterministic=False
        return convert1dAction(env, betaPolicy1d(x, Theta, deterministic=True))

    render_env(env_name, featurizer, Theta, policy_func)