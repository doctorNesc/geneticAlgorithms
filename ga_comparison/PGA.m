% Cellular type PGA vs PGA vs SGA
addpath("genetic/");
clear; clc;

numgen = 1000;       % number of generations
lpop = 450;         % total population size (must be a perfect square for CPGA)
lstring = 10;       % number of genes (10-variable Eggholder)
M = 500;            % maximum of the search space
runs = 5;           % number of independent runs for averaging

rows = 50;
cols = 9;

num_islands = 9;
pop_per_island = 50;

Space = [ones(1, lstring)*(-M); ones(1, lstring)*M];
Delta = Space(2,:) / 100;

sga_evolution_all = zeros(runs, numgen);
cpga_evolution_all = zeros(runs, numgen);
ipga_evolution_all = zeros(runs, numgen);

for run = 1:runs
    fprintf('Execution Run: %d / %d\n', run, runs);
    
    InitialPop = genrpop(lpop, Space);
    
    %% ================== SGA EXECUTION ==================
    Pop_SGA = InitialPop;
    Fit_SGA = eggholder(Pop_SGA);
    
    for gen = 1:numgen
        sga_evolution_all(run, gen) = min(Fit_SGA);
        
        Best  = selbest(Pop_SGA, Fit_SGA, [1,1]); % 2 individuals
        Old   = selrand(Pop_SGA, Fit_SGA, 48);    % 48 individuals
        Work1 = selsus(Pop_SGA, Fit_SGA, 200);    % 200 individuals
        Work2 = selsus(Pop_SGA, Fit_SGA, 200);    % 200 individuals        

        Work1 = crossov(Work1, 1, 0);
        Work2 = mutx(Work2, 0.15, Space);
        Work2 = muta(Work2, 0.15, Delta, Space);
        
        Pop_SGA = [Best; Old; Work1; Work2];
        Fit_SGA = eggholder(Pop_SGA);
    end
    fprintf("SGA minimum on run %d: %.4f\n", run, min(Fit_SGA));
    %% ================== CPGA EXECUTION ==================
    Pop_CPGA = InitialPop;
    Fit_CPGA = eggholder(Pop_CPGA);
    
    for gen = 1:numgen
        cpga_evolution_all(run, gen) = min(Fit_CPGA);
        NewPop_CPGA = zeros(lpop, lstring);
        
        for r = 1:rows
            for c = 1:cols
                idx_c = (r-1)*cols + c;
                
                idx_u = (mod(r-2, rows))*cols + c;
                idx_d = (mod(r, rows))*cols + c;
                idx_l = (r-1)*cols + mod(c-2, cols) + 1;
                idx_r = (r-1)*cols + mod(c, cols) + 1;
                
                neighborhood_idx = [idx_c, idx_u, idx_d, idx_l, idx_r];
                
                N_fit = Fit_CPGA(neighborhood_idx);
                N_pop = Pop_CPGA(neighborhood_idx, :);
                
                t1 = randperm(5, 2); 
                if N_fit(t1(1)) < N_fit(t1(2))
                    p1 = N_pop(t1(1), :);
                else
                    p1 = N_pop(t1(2), :);
                end
                
                t2 = randperm(5, 2);
                if N_fit(t2(1)) < N_fit(t2(2))
                    p2 = N_pop(t2(1), :);
                else
                    p2 = N_pop(t2(2), :);
                end
                
                Parents = [p1; p2];
                
                Offspring = crossov(Parents, 1, 0);
                O1 = Offspring(1, :); 
                
                O1 = mutx(O1, 0.30, Space);
                O1 = muta(O1, 0.30, Delta, Space);
                
                NewPop_CPGA(idx_c, :) = O1;
            end
        end
        NewFit_CPGA = eggholder(NewPop_CPGA);
        
        % Local Elitist Replacement: Offspring replaces parent only if better
        better_idx = NewFit_CPGA < Fit_CPGA; 
        Pop_CPGA(better_idx, :) = NewPop_CPGA(better_idx, :);
        Fit_CPGA(better_idx) = NewFit_CPGA(better_idx);
    end
    fprintf("CPGA minimum on run %d: %.4f\n", run,min(Fit_CPGA));
    
%% ================== ISLAND PGA EXECUTION ==================
    % Initialize 9 distinct islands
    Pop_IPGA = cell(1, num_islands);
    Fit_IPGA = cell(1, num_islands);
    
    stagnation_counter = zeros(1, num_islands);
    best_fit_ever = inf(1, num_islands);
    
    for i = 1:num_islands
        start_idx = (i-1)*pop_per_island + 1;
        end_idx = i*pop_per_island;
        Pop_IPGA{i} = InitialPop(start_idx:end_idx, :);
        Fit_IPGA{i} = eggholder(Pop_IPGA{i});
        best_fit_ever(i) = min(Fit_IPGA{i}); % Initialize best fit
    end

    for gen = 1:numgen
        current_global_min = inf;
        for i = 1:num_islands
            current_global_min = min(current_global_min, min(Fit_IPGA{i}));
        end
        ipga_evolution_all(run, gen) = current_global_min;
        
        % 1. Standard GA Evolution within each island independently
        for i = 1:num_islands
            Best  = selbest(Pop_IPGA{i}, Fit_IPGA{i}, [1,1]); 
            Old   = selrand(Pop_IPGA{i}, Fit_IPGA{i}, 8);     
            Work1 = selsus(Pop_IPGA{i}, Fit_IPGA{i}, 20);     
            Work2 = selsus(Pop_IPGA{i}, Fit_IPGA{i}, 20);     
            
            Work1 = crossov(Work1, 1, 0);
            Work2 = mutx(Work2, 0.15, Space);
            Work2 = muta(Work2, 0.15, Delta, Space);
            
            Pop_IPGA{i} = [Best; Old; Work1; Work2];
            Fit_IPGA{i} = eggholder(Pop_IPGA{i});
            
            current_min = min(Fit_IPGA{i});
            
            % Check if the minimum improved
            if current_min < best_fit_ever(i)
                best_fit_ever(i) = current_min;
                stagnation_counter(i) = 0; % Reset counter on improvement
            else
                stagnation_counter(i) = stagnation_counter(i) + 1;
            end
            
            % Trigger restart if stuck for 100 generations
            if stagnation_counter(i) >= 100
                % Find and preserve the absolute best elite individual
                best_idx = find(Fit_IPGA{i} == current_min, 1);
                elite_ind = Pop_IPGA{i}(best_idx, :);
                elite_fit = Fit_IPGA{i}(best_idx);
                
                % Generate a brand new random population for the island
                Pop_IPGA{i} = genrpop(pop_per_island, Space);
                Fit_IPGA{i} = eggholder(Pop_IPGA{i});
                
                % Inject the preserved elite back into position 1
                Pop_IPGA{i}(1, :) = elite_ind;
                Fit_IPGA{i}(1) = elite_fit;
                
                stagnation_counter(i) = 0; % Reset counter after restart
            end
        end
        
        % 2. Topology-based Migration (Every 100 Generations)
        if mod(gen, 100) == 0
            New_Pop_IPGA = Pop_IPGA;
            New_Fit_IPGA = Fit_IPGA;
            
            for i = 1:num_islands
                r = ceil(i / 3);
                c = mod(i-1, 3) + 1;
                
                neighbours = [];
                
                % STRICT 3x3 GRID (No Torus/Wrap-around)
                if r > 1 % Has a North neighbor
                    neighbours(end+1) = (r-2)*3 + c; 
                end
                if r < 3 % Has a South neighbor
                    neighbours(end+1) = (r)*3 + c;   
                end
                if c > 1 % Has a West neighbor
                    neighbours(end+1) = (r-1)*3 + c - 1; 
                end
                if c < 3 % Has an East neighbor
                    neighbours(end+1) = (r-1)*3 + c + 1; 
                end
                
                % Extract the 2 best individuals from the current island
                [~, sorted_idx] = sort(Fit_IPGA{i});
                migrants = Pop_IPGA{i}(sorted_idx(1:2), :);
                migrants_fit = Fit_IPGA{i}(sorted_idx(1:2));
                
                for n = 1:length(neighbours)
                    dest_island = neighbours(n);
                    
                    % Select 2 random individuals to overwrite in the destination
                    replace_idx = randperm(pop_per_island, 2);
                    
                    New_Pop_IPGA{dest_island}(replace_idx, :) = migrants;
                    New_Fit_IPGA{dest_island}(replace_idx) = migrants_fit;
                end
            end
            Pop_IPGA = New_Pop_IPGA;
            Fit_IPGA = New_Fit_IPGA;
        end
    end
    fprintf("IPGA minimum on run %d: %.4f\n\n", run, current_global_min);end

% Calculate averages over the 5 runs
sga_avg = mean(sga_evolution_all, 1);
cpga_avg = mean(cpga_evolution_all, 1);
ipga_avg = mean(ipga_evolution_all, 1);

figure; hold on; grid on;
plot(1:numgen, sga_avg, 'r', 'LineWidth', 1.5);
plot(1:numgen, cpga_avg, 'b', 'LineWidth', 1.5);
plot(1:numgen, ipga_avg, 'g', 'LineWidth', 1.5);
title('SGA vs. CPGA vs IPGA (Average of 5 Runs)');
xlabel('Generation');
ylabel('Fitness (Average Global Minimum)');
legend('SGA (Panmictic)', 'CPGA (Grid)', 'IPGA (Islands)');