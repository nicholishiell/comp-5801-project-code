import numpy as np
import matplotlib.pyplot as plt

# Define grid
x = np.linspace(-5, 5, 30)
y = np.linspace(-5, 5, 30)
X, Y = np.meshgrid(x, y)

# Point source location
x0, y0 = 0, 0

# Compute distance from source
R = (X - x0)**2 + (Y - y0)**2

# Avoid division by zero at the source
R[R == 0] = 1e-6

# Compute potential field (e.g., gravitational or electrostatic)
C = 1  # Arbitrary constant
Phi = C / R

# Compute gradient (force field components)
Fx, Fy = np.gradient(-Phi, x, y)

# Plot potential field
plt.figure(figsize=(7, 6))
contour = plt.contourf(X, Y, Phi, levels=50, cmap='inferno')
plt.colorbar(label="Potential")

# Plot vector field
plt.quiver(X, Y, -Fy, -Fx, color="white", alpha=0.8, scale=50)

# Mark the point source
plt.scatter(x0, y0, color="cyan", marker="o", label="Point Source")

# Labels and title
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Potential and Vector Field of a Point Source")
plt.legend()
plt.show()