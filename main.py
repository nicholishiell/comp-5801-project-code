import os

import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from SwarmEnv import *
from LearningAlgo import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_temp_field(env):

    # Plot the temperature field and gradient vectors
    plt.figure(figsize=(8, 6))
    plt.contourf(env.get_x_mesh(),
                 env.get_y_mesh(),
                 env.get_temp_map(),
                 cmap='hot',
                 levels=np.linspace(0,10,5),
                 vmin=0.,
                 vmax=50.)
    plt.colorbar(label="Temperature")


    # Plot vector field (gradient)
    grad_y, grad_x = env.get_gradients()
    x_mesh = env.get_x_mesh()
    y_mesh = env.get_y_mesh()

    # Display only every 10th arrow
    plt.quiver(x_mesh[::50, ::50],
               y_mesh[::50, ::50],
               grad_x[::50, ::50],
               grad_y[::50, ::50],
               color='blue',
               scale=50)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Temperature Field and Heat Gradient")
    # plt.show()

    frame_id = f"{env.step_counter:03d}"
    plt.savefig(f"output/temp-map-{frame_id}.png")

    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def policy_fixed(   state : np.ndarray,
                    action_space )->np.ndarray:

    action = np.zeros(action_space.shape, dtype=float)

    for i, row in enumerate(state):

        bearing = row[0]
        intensity = row[1]

        turn = None
        
        if intensity < 20:
            if bearing < 0.:
                turn = 0.25
            elif bearing > 0.:
                turn = 0.75
            else:
                turn = 0.5
        else:
            if bearing > 0.:
                turn = 0.25
            elif bearing < 0.:
                turn = 0.75
            else:
                turn = 0.5

        action[i, 0] = turn

    return action

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def policy_nadda(   state : np.ndarray,
                    action_space ) -> np.ndarray:

    # create empy action array
    action = np.zeros(action_space.shape, dtype=float)

    #Set the values of the first column of action to a random value between -0.1 and 0.1
    action[:,0] = np.random.uniform(0.,
                                    1.,
                                    size=action.shape[0])

    return action

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def run_example(env):

    done = False
    trunc = False
    state, _ = env.reset()

    np.set_printoptions(precision=5, suppress=True)

    while not (done or trunc):

        action = policy_fixed(state, env.action_space)
        # action = policy_nadda(state, env.action_space)
        state, reward, done, trunc, info = env.step(action)

        print(f"Action: {action}")
        print(f"Observation: {state}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        print(f'Step: {env.step_counter} / {env.max_steps}')

        # Plot the temperature field and gradient vectors
        plot_temp_field(env)
        env.render()

    if trunc:
        print("Episode truncated")
    elif done:
        print("Episode done")


    # plot_temp_field(env)

    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():

    #run_training = False
    run_training = True

    env = SwarmEnv( n_agents=5,
                    agent_radius=0.4,
                    max_steps=500,
                    field_size=50.)

    featurizer = RbfSwarmFeaturizer(env, 100)

    if run_training:
        # Train the model
        Theta, w, eval_returns =    ActorCriticCont(env,
                                                    featurizer,
                                                    max_episodes=500,
                                                    evaluate_every=10)
        # Save the model
        np.savez("output/model.npz", Theta=Theta, w=w)
    else:
        run_example(env)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    # Create 'output' directory if it does not exist
    if not os.path.exists('output'):
        os.makedirs('output')

    main()