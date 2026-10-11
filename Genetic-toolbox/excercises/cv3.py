import copy
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# Import the genetic toolbox functions
from genetic_toolbox import genrpop, selbest, selrand, selsus, crossov, mutx, muta, eggholder
# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------
numgen = 1000       # number of generations
lpop = 450          # total population size (must be a perfect square for CPGA)
lstring = 10        # number of genes (10-variable Eggholder)
m = 500             # maximum of the search spacet
runs = 5            # number of independent runs for averaging
rows = 50
cols = 9
num_islands = 9
pop_per_island = 50

# Space definitions: Row 0 -> lower bounds, Row 1 -> upper bounds
Space = np.vstack([np.ones(lstring) * (-m), np.ones(lstring) * m])
Delta = Space[1, :] / 100.0

# Pre-allocate evolution logs
sga_evolution_all = np.zeros((runs, numgen))
cpga_evolution_all = np.zeros((runs, numgen))
ipga_evolution_all = np.zeros((runs, numgen))

# -------------------------------------------------------------------------
# Main Execution Loop
# -------------------------------------------------------------------------
for run in range(runs):
    print(f"Execution Run: {run + 1} / {runs}")
    
    InitialPop = genrpop(lpop, Space)
    
    ## ================== SGA EXECUTION ==================
    Pop_SGA = np.copy(InitialPop)
    Fit_SGA = eggholder(Pop_SGA)
    
    for gen in range(numgen):
        sga_evolution_all[run, gen] = np.min(Fit_SGA)
        
        # EXTRACT THE FIRST ELEMENT [0] FROM SELECTION TUPLES
        Best  = selbest(Pop_SGA, Fit_SGA, [1, 1])[0]   
        Old   = selrand(Pop_SGA, Fit_SGA, 48)[0]       
        Work1 = selsus(Pop_SGA, Fit_SGA, 200)[0]       
        Work2 = selsus(Pop_SGA, Fit_SGA, 200)[0]             
        
        Work1 = crossov(Work1, 1, 0)
        Work2 = mutx(Work2, 0.15, Space)
        Work2 = muta(Work2, 0.15, Delta, Space)
        
        Pop_SGA = np.vstack([Best, Old, Work1, Work2])
        Fit_SGA = eggholder(Pop_SGA)
        
    print(f"SGA minimum on run {run + 1}: {np.min(Fit_SGA):.4f}")
    
    ## ================== CPGA EXECUTION ==================
    Pop_CPGA = np.copy(InitialPop)
    Fit_CPGA = eggholder(Pop_CPGA)
    
    for gen in range(numgen):
        cpga_evolution_all[run, gen] = np.min(Fit_CPGA)
        NewPop_CPGA = np.zeros((lpop, lstring))
        
        for r in range(rows):
            for c in range(cols):
                idx_c = r * cols + c
                idx_u = ((r - 1) % rows) * cols + c
                idx_d = ((r + 1) % rows) * cols + c
                idx_l = r * cols + ((c - 1) % cols)
                idx_r = r * cols + ((c + 1) % cols)
                
                neighborhood_idx = [idx_c, idx_u, idx_d, idx_l, idx_r]
                
                N_fit = Fit_CPGA[neighborhood_idx]
                N_pop = Pop_CPGA[neighborhood_idx, :]
                
                t1 = np.random.choice(5, 2, replace=False)
                p1 = N_pop[t1[0], :] if N_fit[t1[0]] < N_fit[t1[1]] else N_pop[t1[1], :]
                
                t2 = np.random.choice(5, 2, replace=False)
                p2 = N_pop[t2[0], :] if N_fit[t2[0]] < N_fit[t2[1]] else N_pop[t2[1], :]
                
                Parents = np.vstack([p1, p2])
                
                Offspring = crossov(Parents, 1, 0)
                O1 = np.array([Offspring[0, :]]) 
                
                O1 = mutx(O1, 0.30, Space)
                O1 = muta(O1, 0.30, Delta, Space)
                
                NewPop_CPGA[idx_c, :] = O1[0, :]
                
        NewFit_CPGA = eggholder(NewPop_CPGA)
        
        better_idx = NewFit_CPGA < Fit_CPGA 
        Pop_CPGA[better_idx, :] = NewPop_CPGA[better_idx, :]
        Fit_CPGA[better_idx] = NewFit_CPGA[better_idx]
        
    print(f"CPGA minimum on run {run + 1}: {np.min(Fit_CPGA):.4f}")
    
    ## ================== ISLAND PGA EXECUTION ==================
    Pop_IPGA = []
    Fit_IPGA = []
    
    stagnation_counter = np.zeros(num_islands)
    best_fit_ever = np.full(num_islands, np.inf)
    
    for i in range(num_islands):
        start_idx = i * pop_per_island
        end_idx = (i + 1) * pop_per_island
        Pop_IPGA.append(np.copy(InitialPop[start_idx:end_idx, :]))
        Fit_IPGA.append(eggholder(Pop_IPGA[i]))
        best_fit_ever[i] = np.min(Fit_IPGA[i])
        
    for gen in range(numgen):
        current_global_min = np.min([np.min(fit) for fit in Fit_IPGA])
        ipga_evolution_all[run, gen] = current_global_min
        
        for i in range(num_islands):
            # EXTRACT THE FIRST ELEMENT [0] FROM SELECTION TUPLES
            Best  = selbest(Pop_IPGA[i], Fit_IPGA[i], [1, 1])[0] 
            Old   = selrand(Pop_IPGA[i], Fit_IPGA[i], 8)[0]     
            Work1 = selsus(Pop_IPGA[i], Fit_IPGA[i], 20)[0]     
            Work2 = selsus(Pop_IPGA[i], Fit_IPGA[i], 20)[0]     
            
            Work1 = crossov(Work1, 1, 0)
            Work2 = mutx(Work2, 0.15, Space)
            Work2 = muta(Work2, 0.15, Delta, Space)
            
            Pop_IPGA[i] = np.vstack([Best, Old, Work1, Work2])
            Fit_IPGA[i] = eggholder(Pop_IPGA[i])
            
            current_min = np.min(Fit_IPGA[i])
            
            if current_min < best_fit_ever[i]:
                best_fit_ever[i] = current_min
                stagnation_counter[i] = 0 
            else:
                stagnation_counter[i] += 1
            
            if stagnation_counter[i] >= 100:
                best_idx = np.argmin(Fit_IPGA[i])
                elite_ind = np.copy(Pop_IPGA[i][best_idx, :])
                elite_fit = Fit_IPGA[i][best_idx]
                
                Pop_IPGA[i] = genrpop(pop_per_island, Space)
                Fit_IPGA[i] = eggholder(Pop_IPGA[i])
                
                Pop_IPGA[i][0, :] = elite_ind
                Fit_IPGA[i][0] = elite_fit
                
                stagnation_counter[i] = 0 
                
        if (gen + 1) % 100 == 0:
            New_Pop_IPGA = copy.deepcopy(Pop_IPGA)
            New_Fit_IPGA = copy.deepcopy(Fit_IPGA)
            
            for i in range(num_islands):
                r = i // 3
                c = i % 3
                
                neighbours = []
                
                if r > 0: neighbours.append((r - 1) * 3 + c) 
                if r < 2: neighbours.append((r + 1) * 3 + c)   
                if c > 0: neighbours.append(r * 3 + c - 1) 
                if c < 2: neighbours.append(r * 3 + c + 1) 
                
                sorted_idx = np.argsort(Fit_IPGA[i])
                migrants = Pop_IPGA[i][sorted_idx[0:2], :]
                migrants_fit = Fit_IPGA[i][sorted_idx[0:2]]
                
                for dest_island in neighbours:
                    replace_idx = np.random.choice(pop_per_island, 2, replace=False)
                    New_Pop_IPGA[dest_island][replace_idx, :] = migrants
                    New_Fit_IPGA[dest_island][replace_idx] = migrants_fit
                    
            Pop_IPGA = New_Pop_IPGA
            Fit_IPGA = New_Fit_IPGA
            
    print(f"IPGA minimum on run {run + 1}: {current_global_min:.4f}\n")

sga_avg = np.mean(sga_evolution_all, axis=0)
cpga_avg = np.mean(cpga_evolution_all, axis=0)
ipga_avg = np.mean(ipga_evolution_all, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(range(1, numgen + 1), sga_avg, 'r', linewidth=1.5, label='SGA (Panmictic)')
plt.plot(range(1, numgen + 1), cpga_avg, 'b', linewidth=1.5, label='CPGA (Grid)')
plt.plot(range(1, numgen + 1), ipga_avg, 'g', linewidth=1.5, label='IPGA (Islands)')

plt.title('SGA vs. CPGA vs IPGA (Average of 5 Runs)')
plt.xlabel('Generation')
plt.ylabel('Fitness (Average Global Minimum)')
plt.legend()
plt.grid(True)
plt.show()