# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import interpolate

# # Define grid
# x = np.linspace(-5, 5, 50)
# y = np.linspace(-5, 5, 50)
# X, Y = np.meshgrid(x, y)

# # Define heat sources (position and intensity)
# heat_sources = [
#     {"pos": (-2, -2), "intensity": 1},
#     {"pos": (2, 2), "intensity": 1},
#     {"pos": (-3, 3), "intensity": 1}
# ]

# # Compute temperature field
# T = np.zeros_like(X)
# for source in heat_sources:
#     xs, ys = source["pos"]
#     intensity = source["intensity"]
#     distance = np.sqrt((X - xs)**2 + (Y - ys)**2)
#     T += intensity / (distance + 0.1)  # Avoid division by zero


# distance = np.sqrt((X - (-2))**2 + (Y - (-2))**2)
# T -= intensity / (distance + 0.1)  # Avoid division by zero


# temp_at_pos = interpolate.griddata((X.flatten(), Y.flatten()), 
#                                     T.flatten(), 
#                                     (0., 0.), 
#                                     method='cubic')

# print(temp_at_pos)


# # # Compute gradient (vector field)
# # Ty, Tx = np.gradient(T, y, x)  # Gradient components

# # # Normalize the gradient vectors
# # magnitude = np.sqrt(Tx**2 + Ty**2)
# # Tx = Tx / (magnitude + 1e-10)  # Add small value to avoid division by zero
# # Ty = Ty / (magnitude + 1e-10)

# # # Plot the temperature field and gradient vectors
# # plt.figure(figsize=(8, 6))
# # plt.contourf(X, Y, T, cmap='hot', levels=50)
# # plt.colorbar(label="Temperature")

# # # Plot vector field (gradient)
# # plt.quiver(X, Y, Tx, Ty, color='blue', scale=50)  

# # plt.xlabel("X")
# # plt.ylabel("Y")
# # plt.title("Temperature Field and Heat Gradient")
# # plt.show()
import numpy as np
import matplotlib.pyplot as plt
 
 
# Creating arrow
x_pos = 0
y_pos = 0
x_direct = -1
y_direct = 1
 
# Creating plot
fig, ax = plt.subplots(figsize = (12, 7))
ax.quiver(x_pos, y_pos, x_direct, y_direct)
ax.set_title('Quiver plot with one arrow')
 
# Show plot
plt.show()