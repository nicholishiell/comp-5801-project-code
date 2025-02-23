import numpy as np
import matplotlib.pyplot as plt

from SwarmEnv import SwarmEnv

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_temp_field(env):
    # Normalize the gradient vectors
    # magnitude = np.sqrt(grad_x**2 + grad_y**2)
    # grad_x = grad_x / (magnitude + 1e-10)  # Add small value to avoid division by zero
    # grad_y = grad_y / (magnitude + 1e-10)
   
    # Plot the temperature field and gradient vectors
    plt.figure(figsize=(8, 6))
    plt.contourf(env.get_x_mesh(), 
                 env.get_y_mesh(), 
                 env.get_temp_map(), 
                 cmap='hot', 
                 levels=50)
    plt.colorbar(label="Temperature")

    # Plot vector field (gradient)
    grad_y, grad_x = env.get_gradients()
    plt.quiver(env.get_x_mesh(), 
               env.get_y_mesh(), 
               grad_x, 
               grad_y, 
               color='blue', 
               scale=50)  

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Temperature Field and Heat Gradient")
    # plt.show()
    
    frame_id = f"{env.step_counter:03d}"
    plt.savefig(f"output/{frame_id}.png")
    
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def policy(state : np.ndarray,
           action_space )->np.ndarray:

    action = np.zeros(action_space.shape, dtype=float)

    for i, row in enumerate(state):
        
        concentration = row[2]

        gradient = row[:2] 
        gradient = gradient[::-1]

        # if values is less than one travel a long the gradient
        if concentration < 2.:
            action[i] = gradient       
        # if the value is greater than than 2 travel perpendicular to the gradient
        elif concentration < 3.:
            action[i] = np.array([gradient[1], -gradient[0]])            
        # if the value is greater than 2 travel away from the gradient
        else:
            action[i] = -gradient[:2]
    
    return action

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():
    
    
    env = SwarmEnv( n_agents=5,
                    agent_radius=0.4,
                    max_steps=100)
    
    print(f'Observation space: {env.observation_space.shape}')
    print(f'Action space: {env.action_space.shape}')
    
    done = False
    state, _ = env.reset()
    np.set_printoptions(precision=5, suppress=True)
    
    while not done:
        
        action = policy(state, env.action_space)
        state, reward, done, info = env.step(action)
       
        print(f"Action: {action}")
        print(f"Observation: {state}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        
        # Plot the temperature field and gradient vectors
        # plot_temp_field(env)
        env.render()
   
       
    # # plot_temp_field(env)
      

        
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    main()