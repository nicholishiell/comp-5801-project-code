import matplotlib.pyplot as plt
from scipy.special import beta
import numpy as np


def _beta(x, a, b):
    """
    Beta distribution PDF
    """
    return (x**(a-1) * (1-x)**(b-1)) / beta(a, b)

x = np.linspace(0, 1, 1000)
a = 5.826746940612793
b =  4.741579055786133
pdf = _beta(x, a,b)

# file in the output directory
plt.plot(x, pdf, label=f'Beta({a}, {b})')
plt.legend()
plt.show()