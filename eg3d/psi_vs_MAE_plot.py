import matplotlib.pyplot as plt
import numpy as np
from plot_training_results import plot_setup
plot_setup()

plt.figure(figsize=(3,2.5), dpi=300)

plt.grid()
x = [0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
y = [9.086, 5.7969, 5.0344, 4.997, 5.042, 5.196, 5.401, 5.777, 6.095, 6.3903]
#0, 19.032
plt.scatter(x,y,zorder=20)
plt.xlabel(r"$\psi$")
plt.ylabel(r'$\textrm{MAE}_{\texttt{FPAge}}$')

plt.savefig("psi_vs_MAE.png",bbox_inches='tight')
plt.savefig("psi_vs_MAE.pgf", bbox_inches='tight')