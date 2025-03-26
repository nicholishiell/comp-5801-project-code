import numpy as np

# ================================================================================

class Agent:


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(   self,
                    x : float,
                    y : float,
                    heading : float,
                    intensity : float):
        
        # the position of the agent
        self.x = x
        self.y = y
        
        # the velocity of the agent
        self.heading = heading
        
        # the intensity of "pheronome" that the agent is emitting
        self.intensity = intensity
 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def randomize_state(self,
                        x_range: tuple,
                        y_range: tuple,
                        heading_range: tuple,
                        intensity_range: tuple):
        
        self.x = np.random.uniform(x_range[0], x_range[1])
        self.y = np.random.uniform(y_range[0], y_range[1])
        self.heading = np.random.uniform(heading_range[0], heading_range[1])
        self.intensity = np.random.uniform(intensity_range[0], intensity_range[1])
            
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def update_position(self,
                        delta_time: float):

        self.x += np.cos(self.heading) * delta_time
        self.y += np.sin(self.heading) * delta_time
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          
    def update_heading( self,
                        delta_heading: float):
        
        self.heading += delta_heading
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
    def get_position(self)->np.ndarray:
        
        return np.array([self.x, self.y])
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_intensity(self)->float:
        
        return self.intensity
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def set_intensity(self,
                      intensity: float):
        
        self.intensity = intensity

# ================================================================================