import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from genetic_toolbox import *

dim = 10  # number of genes (10-variable Sheffel fn)
lpop = 50  # total population size

swarm_size = 50  # Wing size (particles)
numgen = 500  # number of generations
m = 500  # bounds
runs = 5  # number of independent runs

# PSO Hyperparameters
w = 0.7  # Inertia weight (momentum)
c1 = 2.0
c2 = 1.0

Space = np.vstack([np.ones(dim) * (-m), np.ones(dim) * m])
Delta = Space[1, :] / 100.0

pso_evolution_all = np.zeros((runs, numgen))
ga_evolution = np.zeros((runs, numgen))

for nrun in range(runs):
    print(f"\n--- RUN {nrun + 1} ---")

    ##############################  PSO   ######################################
    pso_pop = genrpop(lpop, Space)
    v_space = uniform_space(dim, -10.0, 10.0)  # velocities
    V = genrpop(swarm_size, v_space)

    # global and local best before the generation loop
    pbest = np.copy(pso_pop)  # Memory of best positions (50x10)
    pbest_fit = testfn3c(pso_pop)  # Memory of best fitnesses (50,)

    best_idx = np.argmin(pbest_fit)
    gbest = np.copy(pbest[best_idx])  # Global best position (10,)
    gbest_fit = pbest_fit[best_idx]  # Global best fitness score (scalar)

    # Loop
    for gen in range(numgen):
        pso_evolution_all[nrun, gen] = gbest_fit

        # random stochastic factors
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)

        # velocity update
        V = (w * V) + (c1 * r1 * (pbest - pso_pop)) + (c2 * r2 * (gbest - pso_pop))

        # Position update
        pso_pop = np.clip(pso_pop + V, Space[0], Space[1])

        # Evaluate newly moved population
        current_fit = testfn3c(pso_pop)

        # Update Personal Bests (only where the new spot is better)
        better_mask = current_fit < pbest_fit
        pbest[better_mask] = pso_pop[better_mask]  # Update coordinates
        pbest_fit[better_mask] = current_fit[better_mask]  # Update scores

        # Update Global Best (if anyone beat the all-time record)
        min_fit_idx = np.argmin(pbest_fit)
        if pbest_fit[min_fit_idx] < gbest_fit:
            gbest = np.copy(pbest[min_fit_idx])
            gbest_fit = pbest_fit[min_fit_idx]

    print(f"PSO | Optimal Fitness Found: {gbest_fit:.4f}")

    ##############################   GA   ######################################
    ga_pop = genrpop(lpop, Space)

    for gen in range(numgen):
        Fit = testfn3c(ga_pop)
        ga_evolution[nrun, gen] = np.min(Fit)

        # selection
        Best, _ = selbest(ga_pop, Fit, [1, 1])
        Old, _ = selrand(ga_pop, Fit, 10)
        Work1, _ = selsus(ga_pop, Fit, 18)
        Work2, _ = selsus(ga_pop, Fit, 20)

        # crossover
        Work1 = crossov(Work1, 1, 0)

        # mutation
        Work2 = mutx(Work2, 0.15, Space)
        Work2 = muta(Work2, 0.15, Delta, Space)

        # Rebuild Population
        ga_pop = np.vstack((Best, Old, Work1, Work2))

    # Final check for GA
    Final_Fit = testfn3c(ga_pop)
    best_ga_fit = np.min(Final_Fit)
    print(f"GA  | Optimal Fitness Found: {best_ga_fit:.4f}")

# --- Plotting Both Averaged Runs ---
pso_avg = np.mean(pso_evolution_all, axis=0)
ga_avg = np.mean(ga_evolution, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(
    range(1, numgen + 1), pso_avg, "r", linewidth=1.5, label="PSO Average (5 runs)"
)
plt.plot(range(1, numgen + 1), ga_avg, "b", linewidth=1.5, label="GA Average (5 runs)")
for i in range(runs):
    plt.plot(
        range(1, numgen + 1),
        pso_evolution_all[i, :],
        linestyle=":",  # Dotted line
        linewidth=0.8,  # Narrow/thin line
        alpha=0.9,  # Slight transparency so overlapping lines blend nicely
        label=f"PSO Run {i + 1}",
    )
plt.title("PSO vs GA Convergence on 10D Sheffel Function")
plt.xlabel("Generation")
plt.ylabel("Average Best Fitness")
plt.legend()
plt.grid(True)
plt.show()
