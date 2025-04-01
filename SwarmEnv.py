import time
import numpy as np
from scipy import interpolate
import pygame

import gymnasium as gym
from gymnasium import spaces
import cv2

from Agent import Agent
from concurrent.futures import ProcessPoolExecutor, as_completed

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MAX_TURN_RATE = 0.1 # Maximum turn rate in radians per time step
TIME_STEP = 0.1 # Time step in seconds
MAX_INTENSITY = 10.0 # Maximum intensity of the signal emitted by an agent

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# this function interpolates the temperature field at a given position
def _interpolate(x,
                    y,
                    map,
                    x_grid,
                    y_grid):

    return interpolate.griddata((x_grid, y_grid),
                                map.flatten(),
                                (x, y),
                                method='cubic')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Calculates the observation for each agent
# loop over all agents to get their individual observations
def process_agent(  i,
                    pos, 
                    heading,
                    temp_map,
                    x_grid,
                    y_grid):
    
    # calculate the gradient of the temperature map
    gx, gy = np.gradient(temp_map)

    c = _interpolate(pos[0],pos[1], temp_map, x_grid, y_grid)
    gx_at_pos = _interpolate(pos[0],pos[1], gx, x_grid, y_grid)
    gy_at_pos = _interpolate(pos[0],pos[1], gy, x_grid, y_grid)

    # calculate the angle of the gradient
    angle_global = np.arctan2(gx_at_pos, gy_at_pos)
    # calculate the angle between the agent's heading and the gradient
    angle_rel = angle_global - heading
    # normalize the angle difference to be between -pi and pi
    angle_rel = (angle_rel + np.pi) % (2 * np.pi) - np.pi

    # calculate intensity
    intensity = c - MAX_INTENSITY  # subtract the intensity produced by the agent
    
    return i, angle_rel, min(intensity, MAX_INTENSITY)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

        # the value of the latest reward
        self.reward = 0

        # number of agents
        self.agent_list = [Agent(0., 0., 0., 0.) for _ in range(n_agents)]

        self.n_agents = n_agents

        # maximum number of steps
        self.max_steps = max_steps

        # store the dimensions of the field
        half_width = 0.5*field_size

        self.x_max =  half_width
        self.y_max =  half_width
        self.x_min = -half_width
        self.y_min = -half_width

        # Define grid
        # np.linespace(min_range, max_range, number_of_points)
        x = np.linspace(-int(half_width), int(half_width), int(half_width*10))
        y = np.linspace(-int(half_width), int(half_width), int(half_width*10))
        self.X, self.Y = np.meshgrid(x, y)

        # The size of the PyGame window (display purposes)
        self.window_size = screen_size
        # radius of the agents (display purposes)
        self.agent_radius = agent_radius
        # remember the field size so we can translate between field and image coordinates
        self.field_size = field_size

        #  Box -    Supports continuous (and discrete) vectors or matrices,
        #           used for vector observations, images, etc
        # the observation space is a 3D vector for each agent
        # the first two values are the gradient of the temperature field
        # the third value is the temperature at the agent's location
        # self.observation_space = spaces.Box(low=np.tile([-1.,-1.,0.], (n_agents, 1)),
        #                                     high=np.tile([1.,1.,float(n_agents)], (n_agents, 1)),
        #                                     shape=(n_agents,3),
        #                                     dtype=np.float32)
        self.observation_space = spaces.Box(low=np.tile([-np.pi,0.], (n_agents, 1)),
                                            high=np.tile([np.pi,float(MAX_INTENSITY)], (n_agents, 1)),
                                            shape=(n_agents,2),
                                            dtype=np.float32)


        # the action space is a 1D vector for each agent
        # only action is the turning rate
        self.action_space = spaces.Box(low=np.tile([-MAX_TURN_RATE], (n_agents, 1)),
                                       high=np.tile([MAX_TURN_RATE], (n_agents, 1)),
                                       shape=(n_agents,1),
                                       dtype=np.float32)

        # Scale factor to convert field coordinates to image coordinates
        self.scale = self.window_size / self.field_size

        # The minimum size of the smallest blob
        # self.min_blob_size = 8028
        self.min_blob_size = 4406

        self.reset()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # loop over all agents and tell them to run their randomize_state method
    def _randomize_agents(self,
                          buffer=0.5):
        for agent in self.agent_list:
            agent.randomize_state((buffer*self.x_min, buffer*self.x_max),
                                  (buffer*self.y_min, buffer*self.y_max),
                                  (-np.pi, np.pi),
                                  (1., 1.))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # This function calculates the temperature field including all sources
    def _calculate_temp_map(self):

        self.is_temp_map_up_to_date = True

        # Compute temperature field
        temp_map = np.zeros_like(self.X)
        for agent in self.agent_list:
            xs = agent.x
            ys = agent.y

            intensity = agent.get_intensity()

            distance_squared = ((self.X - xs)**2 + (self.Y - ys)**2)

            temp_map += intensity / (distance_squared + 1)  # Avoid division by zero

        return temp_map
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    def _get_obs(self):

        # get the temperature map including all agents
        temp_map = self.get_temp_map()

        # create an empty observation array
        observation = np.zeros(self.observation_space.shape, dtype=float)
      
        futures_list = []
      
        with ProcessPoolExecutor() as executor:
            for i, agent in enumerate(self.agent_list):
                future = executor.submit(process_agent, 
                                         i,
                                         agent.get_position(), 
                                         agent.get_heading(),
                                         temp_map, 
                                         self.X.flatten().copy(), 
                                         self.Y.flatten().copy())
                futures_list.append(future)
                
        for future in as_completed(futures_list):
            result = future.result()
            if result is not None:
                i, angle, intensity = result
                
                # store the observation for this agent
                observation[i, 0] = angle
                observation[i, 1] = intensity

        return observation

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # This function is called at the beginning of each episode
    def reset(self,
              seed : int = None,
              options : dict = None):

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # randomly agent positions, headings and intensities
        self._randomize_agents()

        # reset other simulation variables
        self.is_temp_map_up_to_date = False
        self.step_counter = 0

        self.action = np.zeros(self.action_space.shape, dtype=float)

        self.reward = 0

        self.action = None
        observation = self._get_obs()
        info = {}

        return observation, info

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # calculate the reward by finding the largest blob in the image
    def _calculate_reward(self):

        # each agent receives a reward based on the size of the blob it is in
        reward_vector = np.zeros(len(self.agent_list))

        # Create a blank image
        img = np.zeros((self.window_size,
                        self.window_size, 3), dtype=np.uint8)

        for i, agent in enumerate(self.agent_list):
            # skip agents that are not visible
            if not self._agent_visible(agent):
                continue

            # get the position of the agent
            pos = agent.get_position()

            # Convert agent position to image coordinates
            center = (int((pos[0] + self.field_size / 2) * self.scale),
                        int((pos[1] + self.field_size / 2) * self.scale))

            cv2.circle(img, center, int(agent.vision_range * self.scale), (255, 0, 0), -1)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold the image to create a binary image
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate the size of each blob
        blob_sizes = [cv2.contourArea(contour) for contour in contours]

        # Find which contour each agent is inside of
        for i, agent in enumerate(self.agent_list):
            pos = agent.get_position()

            point = (int((pos[0] + self.field_size / 2) * self.scale),
                 int((pos[1] + self.field_size / 2) * self.scale))
            for j, contour in enumerate(contours):
                if cv2.pointPolygonTest(contour, point, False) >= 0:
                    reward_vector[i] = blob_sizes[j] - self.min_blob_size
                    break
            
        return reward_vector

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check if the end of the episode has been reached
    def _check_done(self):
        return self.step_counter >= self.max_steps
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # if an agent is near the edge of the field truncate the episode
    # this is to prevent agents from going out of bounds
    def _check_trunc(self):
        
        buffer = 0.01*self.field_size
        
        for agent in self.agent_list:
                pos = agent.get_position()
                if (pos[0] < self.x_min+buffer or pos[0] > self.x_max-buffer or
                    pos[1] < self.y_min+buffer or pos[1] > self.y_max-buffer):
                    return True
        return False
   
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def _convertAction(self, a):
        """
        Convert an action a in [0, 1] to an actual action
        specified by the the range of the environment.
        Assumes that the action space is 1d.
        """
        a = (a * (self.action_space.high[0] - self.action_space.low[0])
            + self.action_space.low[0])
        return np.array([a, ])              

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # This function is called at each time step
    # the action is a 1D vector for each agent so the shape is (n_agents, 1)
    # the value represents the turning rate of the agent
    
    def step(self, action):
        s_total = time.time()
        # Update the heading of each agent based on the action taken and then update the position
        for i, agent in enumerate(self.agent_list):
            # convert [0,1] action to [-MAX_TURN_RATE, MAX_TURN_RATE]                
            w = self._convertAction(action[i, 0])    
            # apply action to agent
            agent.update_heading(w)
            agent.set_intensity(MAX_INTENSITY)

            # update the position of the agent
            agent.update_position(TIME_STEP)

        # the temperature map is no longer up to date
        self.is_temp_map_up_to_date = False

        # increment the step counter
        self.step_counter += 1

        # Get the observations
        observation = self._get_obs()

        # Calculate the reward
        self.reward = self._calculate_reward()
   
        done = self._check_done()
        trunc = self._check_trunc()
        info = {}

        self.action = action

        return observation, self.reward, done, trunc, info

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check if an agent is within the visible area
    def _agent_visible(self, agent) -> bool:

        return self.x_min <= agent.x <= self.x_max and self.y_min <= agent.y <= self.y_max

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def render(self):
        # Create a blank image
        img = np.zeros((self.window_size,
                        self.window_size, 3), dtype=np.uint8)

        for i, agent in enumerate(self.agent_list):

            # skip agents that are not visible
            if not self._agent_visible(agent):
                continue

            # get the position of the agent
            pos = agent.get_position()

            # Convert agent position to image coordinates
            center = (int((pos[0] + self.field_size / 2) * self.scale),
                        int((pos[1] + self.field_size / 2) * self.scale))

            # draw the circle representing the agent
            cv2.circle(img, center, int(self.agent_radius * self.scale), (255, 0, 0), -1)

            # Draw a dashed green circle around the agent to represent its vision range
            radius = int(agent.vision_range * self.scale)
            color = (0, 255, 0)  # Green color
            thickness = 1  # Thickness of the dashed circle
            num_dashes = 36  # Number of dashes in the circle
            for i in range(num_dashes):
                start_angle = i * (360 / num_dashes)
                end_angle = start_angle + (360 / (2 * num_dashes))
                cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, color, thickness)

        # show the gradient vector observed by the agent
        obs = self._get_obs()
        for i, row in enumerate(obs):
            pos = self.agent_list[i].get_position()
            heading = self.agent_list[i].get_heading()
            grad_x = np.cos(row[0] + heading)
            grad_y = np.sin(row[0] + heading)
            
            # Convert agent position to image coordinates
            x = int((pos[0] + self.field_size / 2) * self.scale)
            y = int((pos[1] + self.field_size / 2) * self.scale)
            # Calculate the end point of the arrow based on the angle relative to heading
            arrow_length = int(self.agent_radius * self.scale * 2)
            end_x = int(x + arrow_length * grad_x)
            end_y = int(y + arrow_length * grad_y)
            # Draw the arrow representing the gradient vector
            cv2.arrowedLine(img, (x, y), (end_x, end_y), (255, 255, 0), 1, tipLength=0.3)


        # show the heading of the agent
        for agent in self.agent_list:
            pos = agent.get_position()
            heading = agent.get_heading()

            # Convert agent position to image coordinates
            x = int((pos[0] + self.field_size / 2) * self.scale)
            y = int((pos[1] + self.field_size / 2) * self.scale)

            # Calculate the end point of the arrow based on the heading
            arrow_length = int(self.agent_radius * self.scale * 2)
            end_x = int(x + arrow_length * np.cos(heading))
            end_y = int(y + arrow_length * np.sin(heading))

            # Draw the arrow representing the agent's heading
            cv2.arrowedLine(img, (x, y), (end_x, end_y), (0, 255, 255), 1, tipLength=0.3)

        # Write the step counter in the top right corner of the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # White color
        thickness = 1
        text = f"Step: {self.step_counter}"
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = self.window_size - text_size[0] - 10  # 10 pixels from the right edge
        text_y = text_size[1] + 10  # 10 pixels from the top edge
        cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)

        largest_blob = np.max(self.reward)
        text = f"Largest blob: {largest_blob:.2f}"
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = self.window_size - text_size[0] - 10
        text_y = text_size[1] + 30
        cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)

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

    def get_y_mesh(self)->np.ndarray:
        return self.Y

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~