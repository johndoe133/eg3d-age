import matplotlib.pyplot as plt
import numpy as np
from plot_training_results import plot_setup
plot_setup()

plt.figure(figsize=(2,1.5), dpi=300)
plt.plot([-1,1], [1,0])
plt.xlabel(r"$S_C(f_i, f_j)$")
plt.ylabel(r"$d_{\cos}$")

plt.savefig("id_loss.png",bbox_inches='tight')
plt.savefig("id_loss.pgf", bbox_inches='tight')

plt.figure(figsize=(2.5,2), dpi=300)
x = np.linspace(-3,3,4000)
y = 0.25 * np.cos((np.pi/2) * x) + 0.75
plt.plot(x,y)

plt.xlim(-2.1, 2.1)
plt.ylim(0.25,1.1)
plt.xlabel(r"$\Delta_{\textrm{age}}$")
plt.ylabel(r"$w(\Delta_{\textrm{age}})$")
plt.savefig("weight_f.png",bbox_inches='tight')
plt.savefig("weight_f.pgf", bbox_inches='tight')