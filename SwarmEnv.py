import numpy as np
from scipy import interpolate
import pygame

import gymnasium as gym
from gymnasium import spaces
import cv2

INTENSITY = 1.

class SwarmEnv(gym.Env):
    
    metadata = {'render.modes': ['human', 'rgb_array'], 
                "render_fps": 4}
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # this function initializes the environment
    def __init__(self,
                 render_mode=None,
                 n_agents=10,
                 agent_radius=0.4,
                 max_steps=1000,
                 field_size = 10.,
                 screen_size = 512):
        
        # number of agents
        self.n_agents = n_agents
        
        # radius of the agents
        self.agent_radius = agent_radius
        
        # maximum number of steps
        self.max_steps = max_steps
        
        # store the dimensions of the field
        self.field_size = field_size
        
        # Initialize the agent locations        
        self.swarm_xyw = np.zeros((n_agents, 3), dtype=float)
       
        # The size of the PyGame window
        self.window_size = screen_size 

        # Define grid
        half_width = field_size / 2.
        x = np.linspace(-int(half_width), int(half_width), int(half_width*10))
        y = np.linspace(-int(half_width), int(half_width), int(half_width*10))
        self.X, self.Y = np.meshgrid(x, y)

        #  Box -    Supports continuous (and discrete) vectors or matrices, 
        #           used for vector observations, images, etc
        self.observation_space = spaces.Box(low=-1.*n_agents,
                                            high=1.*n_agents, 
                                            shape=(self.n_agents,3), 
                                            dtype=np.float32)
        
        self.action_space = spaces.Box(low=-1.0,
                                       high=1.0,
                                       shape=(self.n_agents,3),
                                       dtype=np.float32)
        
        self.reset()
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def _randomize_agent_locations(self):
        half_width = self.field_size / 3
        self.swarm_xyw[:, :2] = np.random.uniform(-half_width, half_width, (self.n_agents, 2))
        self.swarm_xyw[:, 2] = 1.0
      
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # This function calculates the temperature field including all sources
    def _calculate_temp_map(self):
                
        self.is_temp_map_up_to_date = True
                         
        # Compute temperature field
        temp_map = np.zeros_like(self.X)
        for source in self.swarm_xyw:
            xs = source[0]
            ys = source[1]
            
            intensity = source[2]
            distance = ((self.X - xs)**2 + (self.Y - ys)**2)
            temp_map += intensity / (distance + 1e-6)  # Avoid division by zero
                        
        return temp_map
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # this function interpolates the temperature field at a given position
    def _interpolate(self,
                     x,
                     y,
                     map):
        
        return interpolate.griddata((self.X.flatten(), self.Y.flatten()), 
                                    map.flatten(), 
                                    (x, y), 
                                    method='cubic')
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # This function calculates the temperature field at a given position minus the
    # contribution of the target source, using interpolation at the location of the
    # target source
    def _get_obs_per_pos(   self,
                            temp_map, 
                            source):
        
        intensity = source[2]
        distance = ((self.X - source[0])**2 + (self.Y - source[1])**2)
        temp_map -= intensity / (distance + 1e-6)  # Avoid division by zero
        
        # Compute gradient
        gx, gy = np.gradient(temp_map)
              
        # Interpolate temperature map at position pos
        temp_at_pos = self._interpolate(source[0], source[1], temp_map)
        gx_at_pos = self._interpolate(source[0], source[1], gx)
        gy_at_pos = self._interpolate(source[0], source[1], gy)
       
        return temp_at_pos, gx_at_pos, gy_at_pos       
       
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def _get_obs(self):
        
        temp_map = self.get_temp_map()
        
        observation = np.zeros(self.observation_space.shape, dtype=float)
               
        for i,pos in enumerate(self.swarm_xyw):
            c,gx,gy = self._get_obs_per_pos(temp_map.copy(),
                                            pos)  
        
            norm = np.sqrt(gx**2 + gy**2)
            if norm != 0:
                gx /= norm
                gy /= norm
               
            observation[i, 0] = gx
            observation[i, 1] = gy
            observation[i, 2] = c
                
        return observation

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           

    def reset(self,
              seed : int = None,
              options : dict = None):
        
        # We need the following line to seed self.np_random
        super().reset(seed=seed)      
        
        # randomly position agents
        self._randomize_agent_locations()
        
        # reset other simulation variables
        self.is_temp_map_up_to_date = False
        self.step_counter = 0
    
        self.action = np.zeros(self.action_space.shape, dtype=float)
        
        self.action = None
        observation = self._get_obs()
        info = {}
        
        return observation, info
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           
    
    def _calculate_reward(self):
       
        # # Create a blank image
        # img = np.zeros((self.window_size, 
        #                 self.window_size, 3), dtype=np.uint8)
        
        # # Scale factor to convert field coordinates to image coordinates
        # scale = self.window_size / self.field_size
        
        # for agent in self.swarm_xy:
        #     # Convert agent position to image coordinates
        #     center = (int((agent[0] + self.field_size / 2) * scale), 
        #                 int((agent[1] + self.field_size / 2) * scale))
            
        #     # Draw the circle
        #     cv2.circle(img, center, int(self.agent_radius * scale), (255, 0, 0), -1)
                
        
        # # Save the image to a PNG file with a filename related to the current step
        # frame_id = f"{self.step_counter:03d}"
        # cv2.imwrite(f"output/{frame_id}.png")
        
        # # Convert the image to grayscale
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # # Threshold the image to create a binary image
        # _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # # Find contours in the binary image
        # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # # Find the largest contour
        # largest_contour = max(contours, key=cv2.contourArea)
        
        # # Calculate the area of the largest contour
        # largest_blob_size = cv2.contourArea(largest_contour)
        
       
        
        # Display the image
        # cv2.imshow('Swarm Environment', img)
        # cv2.waitKey(5000)
        
        # Placeholder for reward calculation logic
        reward = 0.0
        
        return reward
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def _check_done(self):
        return self.step_counter >= self.max_steps
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def step(self, action):
        
        intensity = action[:, 2]           
        velocity = action[:, :2]
            
        # Clip the action values to the range [-1, 1]
        velocity = np.clip(velocity, -1., 1.)
        
        # Update the agent locations
        self.swarm_xyw[:, :2] += 0.05*velocity        
        
        # update the agents intensity
        self.swarm_xyw[:, 2] = np.clip(intensity,1.,10.)
        
        # Clip the agent locations to the field boundaries
        # half_width = self.field_size / 2
        # self.swarm_xy = np.clip(self.swarm_xy, -half_width, half_width)
        
        self.is_temp_map_up_to_date = False
        self.step_counter += 1
        
        # Get the observations
        observation = self._get_obs()
        
        # Calculate the reward
        reward = self._calculate_reward()
        done = self._check_done()
        info = {}
        
        self.action = action
        
        return observation, reward, done, info

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def render(self):
        # Create a blank image
        img = np.zeros((self.window_size, 
                        self.window_size, 3), dtype=np.uint8)
        
        # Scale factor to convert field coordinates to image coordinates
        scale = self.window_size / self.field_size
        
        for agent in self.swarm_xyw:
            # Convert agent position to image coordinates
            center = (int((agent[0] + self.field_size / 2) * scale), 
                        int((agent[1] + self.field_size / 2) * scale))
            
            # Draw the circle
            cv2.circle(img, center, int(self.agent_radius * scale), (255, 0, 0), -1)
                
        
        # draw a 
        obs = self._get_obs()
        for i, row in enumerate(obs):
            x = int((self.swarm_xyw[i][0] + self.field_size / 2) * scale)
            dy = int(row[0] * 10)
            
            y = int((self.swarm_xyw[i][1] + self.field_size / 2) * scale)
            dx = int(row[1] * 10)
                       
            cv2.arrowedLine(img, (x, y), (x + dx, y + dy), (0, 255, 0), 1)
            
        action = self.action
        for i, row in enumerate(action):
            x = int((self.swarm_xyw[i][0] + self.field_size / 2) * scale)
            dx = int(row[0] * 10)
            
            y = int((self.swarm_xyw[i][1] + self.field_size / 2) * scale)
            dy = int(row[1] * 10)
            
            cv2.arrowedLine(img, (x, y), (x + dx, y + dy), (0, 0, 255), 1)
        
        
        # Save the image to a PNG file with a filename related to the current step
        frame_id = f"{self.step_counter:03d}"
        cv2.imwrite(f"output/{frame_id}.png",img)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def close(self):
        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_temp_map(self)->np.ndarray:
        
        if not self.is_temp_map_up_to_date:
            self.temp_map = self._calculate_temp_map()    
                
        return self.temp_map
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_gradients(self):
        
        temp_map = self.get_temp_map()
        
        gx, gy = np.gradient(temp_map)
        
        norm = np.sqrt(gx**2 + gy**2)
        norm[norm == 0] = 1  # Avoid division by zero
        gx /= norm
        gy /= norm
        
        return gx, gy
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_x_mesh(self)->np.ndarray:
        return self.X 
    
     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_y_mesh(self)->np.ndarray:
        return self.Y
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~