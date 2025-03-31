import numpy as np
from scipy import interpolate
import pygame

import gymnasium as gym
from gymnasium import spaces
import cv2

from Agent import Agent

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MAX_TURN_RATE = 0.1 # Maximum turn rate in radians per time step
TIME_STEP = 0.1 # Time step in seconds
MAX_INTENSITY = 10.0 # Maximum intensity of the signal emitted by an agent

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
        self.observation_space = spaces.Box(low=np.tile([-1.,-1.,0.], (n_agents, 1)),
                                            high=np.tile([1.,1.,float(n_agents)], (n_agents, 1)),
                                            shape=(n_agents,3),
                                            dtype=np.float32)

        # the action space is a 2D vector for each agent
        # the first value is the turning rate
        # the second value is the intensity of the signal emitted by the agent
        self.action_space = spaces.Box(low=np.tile([-MAX_TURN_RATE, .0], (n_agents, 1)),
                                       high=np.tile([MAX_TURN_RATE,1.0], (n_agents, 1)),
                                       shape=(n_agents,2),
                                       dtype=np.float32)

        # Scale factor to convert field coordinates to image coordinates
        self.scale = self.window_size / self.field_size

        # The minimum size of the smallest blob
        self.min_blob_size = 8028

        self.reset()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # loop over all agents and tell them to run their randomize_state method
    def _randomize_agents(self,
                          buffer=0.5):
        for agent in self.agent_list:
            agent.randomize_state((buffer*self.x_min, buffer*self.x_max),
                                  (buffer*self.y_min, buffer*self.y_max),
                                  (-np.pi, np.pi),
                                  (0., 1))

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
    # Calculates the observation for each agent
    def _get_obs(self):

        # get the temperature map including all agents
        temp_map = self.get_temp_map()

        # create an empty observation array
        observation = np.zeros(self.observation_space.shape, dtype=float)

        # loop over all agents to get their individual observations
        for i, agent in enumerate(self.agent_list):

            gx, gy = np.gradient(temp_map)
            c = self._interpolate(agent.x, agent.y, temp_map)
            gx_at_pos = self._interpolate(agent.x, agent.y, gx)
            gy_at_pos = self._interpolate(agent.x, agent.y, gy)

            norm = np.sqrt(gx_at_pos**2 + gy_at_pos**2)
            if norm != 0:
                gx /= norm
                gy /= norm

            observation[i, 0] = gx_at_pos
            observation[i, 1] = gy_at_pos
            observation[i, 2] = c - agent.get_intensity() # subtract the intensity produced by the agent

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
    # This function is called at each time step
    # the action is a 2D vector for each agent so the shape is (n_agents, 2)
    # first value is the turning rate
    # second value is the intensity of the signal emitted by the agent
    def step(self, action):

        turning_rate = action[:, 0]
        intensity = action[:, 1]

        for i, agent in enumerate(self.agent_list):
            # apply action to agent
            agent.update_heading(turning_rate[i])
            agent.set_intensity(intensity[i])

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
        info = {}

        self.action = action

        return observation, self.reward, done, info

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
        # obs = self._get_obs()
        # for i, row in enumerate(obs):
        #     x = int((self.swarm_xyw[i][0] + self.field_size / 2) * self.scale)
        #     dy = int(row[0] * 10)

        #     y = int((self.swarm_xyw[i][1] + self.field_size / 2) * self.scale)
        #     dx = int(row[1] * 10)

        #     cv2.arrowedLine(img, (x, y), (x + dx, y + dy), (0, 255, 0), 1)

        # show the heading of the agent
        # action = self.action
        # for i, row in enumerate(action):
        #     x = int((self.swarm_xyw[i][0] + self.field_size / 2) * self.scale)
        #     dx = int(row[0] * 10)

        #     y = int((self.swarm_xyw[i][1] + self.field_size / 2) * self.scale)
        #     dy = int(row[1] * 10)

        #     cv2.arrowedLine(img, (x, y), (x + dx, y + dy), (0, 0, 255), 1)

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