import matplotlib.pyplot as plt

from plot_training_results import plot_setup
plot_setup()

plt.figure(figsize=(2,1.5), dpi=300)
plt.plot([-1,1], [20,0])
plt.xlabel(r"$S_C(f_i, f_j)$")
plt.ylabel(r"$\mathcal{L}_{ID}$")

plt.savefig("test.png",bbox_inches='tight')
plt.savefig("id_loss.pgf", bbox_inches='tight')