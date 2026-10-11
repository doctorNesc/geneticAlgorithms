import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from genetic_toolbox import *

dim = 10  # number of genes (10-variable Sheffel fn)
lpop = 50  # total population size

swarm_size = 50  # Wing size (particles)
numgen = 1000  # number of generations
m = 500  # bounds
runs = 5  # number of independent runs

# PSO Hyperparameters
w_start = 0.9  # Initial inertia
w_end = 0.4  # Final inertia
c1 = 2.0
c2 = 1.0
v_max = 800.0  # Velocity clamping to prevent overshooting

Space = np.vstack([np.ones(dim) * (-m), np.ones(dim) * m])
Delta = Space[1, :] / 100.0

pso_evolution_all = np.zeros((runs, numgen))
ga_evolution = np.zeros((runs, numgen))

for nrun in range(runs):
    print(f"\n--- RUN {nrun + 1} ---")

    ##############################  PSO   ######################################
    # 1. Initialize Positions and Velocities for this run
    pso_pop = genrpop(lpop, Space)
    v_space = uniform_space(dim, -10.0, 10.0)
    V = genrpop(swarm_size, v_space)

    # 2. Initialize Memory (pbest and gbest)
    pbest = np.copy(pso_pop)
    pbest_fit = testfn3c(pso_pop)

    best_idx = np.argmin(pbest_fit)
    gbest = np.copy(pbest[best_idx])
    gbest_fit = pbest_fit[best_idx]

    # 3. Generation Loop
    for gen in range(numgen):
        # Linear Decreasing Inertia for better convergence
        w = w_start - ((w_start - w_end) * (gen / numgen))

        pso_evolution_all[nrun, gen] = gbest_fit
        c1_dynamic = 2.5 - (2.0 * (gen / numgen))
        c2_dynamic = 0.5 + (2.0 * (gen / numgen))
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)

        V = (
            (w * V)
            + (c1_dynamic * r1 * (pbest - pso_pop))
            + (c2_dynamic * r2 * (gbest - pso_pop))
        )
        V = np.clip(V, -v_max, v_max)  # Velocity clamping prevents explosion
        pso_pop = pso_pop + V
        hit_wall = (pso_pop < Space[0]) | (pso_pop > Space[1])

        # Clip them back inside bounds
        pso_pop = np.clip(pso_pop, Space[0], Space[1])

        # Reverse and dampen the velocity of particles that crashed
        V[hit_wall] *= -0.5

        current_fit = testfn3c(pso_pop)

        better_mask = current_fit < pbest_fit
        pbest[better_mask] = pso_pop[better_mask]
        pbest_fit[better_mask] = current_fit[better_mask]

        min_fit_idx = np.argmin(pbest_fit)
        if pbest_fit[min_fit_idx] < gbest_fit:
            gbest = np.copy(pbest[min_fit_idx])
            gbest_fit = pbest_fit[min_fit_idx]

    print(f"PSO | Optimal Fitness Found: {gbest_fit:.4f}")

    ##############################   GA   ######################################
    # 1. Initialize Population for this run
    ga_pop = genrpop(lpop, Space)

    # 2. Generation Loop
    for gen in range(numgen):
        Fit = testfn3c(ga_pop)
        ga_evolution[nrun, gen] = np.min(Fit)

        # Selection
        Best, _ = selbest(ga_pop, Fit, [1, 1])
        Old, _ = selrand(ga_pop, Fit, 10)
        Work1, _ = selsus(ga_pop, Fit, 18)
        Work2, _ = selsus(ga_pop, Fit, 20)

        # Crossover
        Work1 = crossov(Work1, 1, 0)

        # Mutation
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
print(f"Lowest PSO: {np.min(pso_evolution_all)}")
print(f"Lowest GA: {np.min(ga_evolution)}")

plt.figure(figsize=(10, 6))
plt.plot(
    range(1, numgen + 1), pso_avg, "r", linewidth=1.5, label="PSO Average (5 runs)"
)
plt.plot(range(1, numgen + 1), ga_avg, "b", linewidth=1.5, label="GA Average (5 runs)")

plt.title("PSO vs GA Convergence on 10D Sheffel Function")
plt.xlabel("Generation")
plt.ylabel("Average Best Fitness")
plt.legend()
plt.grid(True)
plt.show()
