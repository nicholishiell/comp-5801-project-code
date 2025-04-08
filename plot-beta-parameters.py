
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import beta


def _beta(x, a, b):
    """
    Beta distribution PDF
    """
    return (x**(a-1) * (1-x)**(b-1)) / beta(a, b)


def softplus(x):
    """
    Softplus function
    """
    return np.log1p(np.exp(x))

def plot_beta(i, a, b) -> None:
    x = np.linspace(0, 1, 1000)

    # Compute the PDF
    pdf = beta.pdf(x, softplus(a)+1., softplus(b)+1.)
    print(softplus(a)+1., softplus(b)+1.)

    # file in the output directory
    output_dir = '/home/nicholishiell/Documents/COMP5801/CourseProject/comp-5801-project-code/output/'
    plt.plot(x, pdf, label=f'Beta({a}, {b})')
    plt.legend()
    plt.savefig(f"{output_dir}beta_plot_{i}.png")
    plt.clf()  # Clear the figure for the next plot

# Load the data from the TSV file
file_path = 'policy-parameters.csv'

with(open(file_path, 'r')) as f:
    lines = f.readlines()

# Plot the data
plt.figure(figsize=(8, 6))
beta_values = []
for i,row in enumerate(lines):
    row = row.strip().split(',')
    # plot_beta(i, float(row[1]), float(row[2]))
    
    color = plt.cm.viridis(int(row[0]) / 100)  # Map row[0] to a color using a colormap
    plt.scatter(float(row[1]), float(row[2]), color=color, label=f"({row[1]}, {row[2]})")

# Add labels and title
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Plot of Policy Parameters')
plt.grid(True)

# Show the plot
plt.show()


