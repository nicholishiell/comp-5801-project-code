import os

import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from SwarmEnv import *
from LearningAlgo import *
import sys

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
    plt.close()
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MAX_TURN_RATE = np.pi/10

def calculate_action( bearing, intensity) -> float:
    
    turn = 0.5
    if intensity < 50:
        if bearing > 0:
            turn = 1.
        elif bearing < 0:
            turn = 0.
    else:
        if bearing > 0:
            turn = 0.
        elif bearing < 0:
            turn = 1.


    return turn

def policy_fixed(   state : np.ndarray,
                    action_space )->np.ndarray:

    action = np.zeros(action_space.shape, dtype=float)


    for i, row in enumerate(state):

        bearing = row[0]
        intensity = row[1]
               
        action[i, :] = calculate_action(bearing, intensity)
        
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
def plot_policy_fixed(policy_function):
            

    ang_points = np.arange(-np.pi, np.pi, 0.1)
    intensity_points = np.arange(0., 100., 0.5)

    # plot the policy
    policy_plot_data = []
    for ang in ang_points:
        for intensity in intensity_points:
            a = policy_function(ang, intensity)
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
    plt.savefig('output/policy_fixed.png')
    plt.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_model(model_path):
    # Load the model parameters from the file
    data = np.load(model_path)
    Theta = data['Theta']
    w = data['w']

    return Theta, w

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def run_learned_policy_example(env, featurizer):
    
    # Load the model
    model_path = "output/model.npz"
    theta, _ = load_model(model_path)
 
    print("Theta:")
    pprint(theta)
 
    done = False
    trunc = False
    state, _ = env.reset()
    
    # calculate the state feature vector
    x_s = featurizer.featurize_swarm(state)
    
    _,_,action_values = plot_policy(theta=theta,featurizer=featurizer,filename="loaded_policy.png")
    
    # print("Action Values:")
    # pprint(action_values)
    
    # while not (done or trunc):
    #     action = get_swarm_actions( x_s, 
    #                                 theta,
    #                                 env.action_space)

    #     state, _, done, trunc, _ = env.step(action)

    #     # Plot the temperature field and gradient vectors
    #     env.render()

    #     # # update the state
    #     x_s = featurizer.featurize_swarm(state)
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def run_fixed_policy_example(env):

    done = False
    trunc = False
    state, _ = env.reset()

    np.set_printoptions(precision=3, suppress=True)

    plot_policy(calculate_action)
    input("Press Enter to continue...")
    while not (done or trunc):

        action = policy_fixed(state, env.action_space)
        state, reward, done, trunc, info = env.step(action)

        print(f"Action: {action}")
        print(f"Observation: {state}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        print(f'Step: {env.step_counter} / {env.max_steps}')
        
        # Plot the temperature field and gradient vectors
        # plot_temp_field(env)
        env.render()


    if trunc:
        print("Episode truncated")
    elif done:
        print("Episode done")


    # plot_temp_field(env)

    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main(run_option):

    env = SwarmEnv( n_agents=5,
                    agent_radius=0.4,
                    max_steps=500,
                    field_size=25.)

    featurizer = RbfSwarmFeaturizer(env, 1000)

    if run_option == "train":
        # Train the model
        Theta, w, eval_returns =    ActorCriticEpisodic(env,
                                                        featurizer,
                                                        max_episodes=100,
                                                        evaluate_every=10)
        # Save the model
        np.savez("output/model.npz", 
                 Theta=Theta, 
                 ft_centers=featurizer._centers, 
                 ft_means=featurizer._mean, 
                 ft_std =featurizer._std) 
        
    elif run_option == "fixed":
        run_fixed_policy_example(env)
    elif run_option == "learned":
        run_learned_policy_example(env, featurizer)
        

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    # Create 'output' directory if it does not exist
    if not os.path.exists('output'):
        os.makedirs('output')

    if len(sys.argv) != 2:
        print("Usage: python main.py <arg>")
        sys.exit(1)

    run_option = sys.argv[1]
              
    main(run_option)