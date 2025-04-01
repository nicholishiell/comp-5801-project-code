import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def function(xc,yc,xp,yp):
    return 10. / ((xc-xp)**2 + (yc-yp)**2 + 1.)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def generate_full_map(x_grid, y_grid,
                      source_a_x, source_a_y,
                      source_b_x, source_b_y):

    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid,indexing='xy')

    full_map = np.zeros_like(x_mesh)

    full_map += 10. / (((x_mesh - source_a_x)**2 + (y_mesh - source_a_y)**2) + 1.)
    full_map += 10. / (((x_mesh - source_b_x)**2 + (y_mesh - source_b_y)**2) + 1.)

    return  full_map
    # map = np.zeros((len(x_grid), len(y_grid)))
    # for i in range(len(x_grid)):
    #     for j in range(len(y_grid)):
    #         x = x_grid[i]
    #         y = y_grid[j]
    #         map[i, j] += function(source_a_x,  source_a_y, x, y) + function(source_b_x,  source_b_y, x, y)
    # return map

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def subtract_from_map(x_grid, y_grid, map, source_x, source_y):

    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid,indexing='xy')
    source_contribution = 10. / (((x_mesh - source_x)**2 + (y_mesh - source_y)**2) + 1.)
    return  map - source_contribution

    # map_copy = np.copy(map)
    # for i in range(len(map)):
    #     for j in range(len(map)):
    #         x = x_grid[i]
    #         y = y_grid[j]
    #         map_copy[i, j] -= function(source_x, source_y, x, y)

    # return map_copy

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():

    n_grid_pts = 100

    x_grid = np.linspace(-5, 5, n_grid_pts)
    y_grid = np.linspace(-5, 5, n_grid_pts)

    source_a_x = 0.
    source_a_y = 1.5

    source_b_x = 0.
    source_b_y = -1.5

    x_query = 0.
    y_query = 1.5

    full_map = generate_full_map(x_grid, y_grid,
                                 source_a_x, source_a_y,
                                 source_b_x, source_b_y)


    map_remove_a = subtract_from_map(x_grid, y_grid, full_map, source_a_x, source_a_y)
    map_remove_b = subtract_from_map(x_grid, y_grid, full_map, source_b_x, source_b_y)

    interpolator = interpolate.RegularGridInterpolator((x_grid, y_grid),
                                                        full_map,
                                                        method='cubic',
                                                        bounds_error=False,
                                                        fill_value=None)


    val = interpolator((x_query, y_query))
    true_val = function(source_a_x,source_a_y, x_query, y_query) + function(source_b_x,source_b_y, x_query, y_query)

    print(f"Interpolated value at ({x_query}, {y_query}) is {val} true value is {true_val}")


    plt.imshow(full_map, extent=(-5, 5, -5, 5), origin='lower', cmap='viridis')
    plt.colorbar(label='Function Value')
    plt.title('Heatmap of Function Values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    main()