import re
import os
import random
from deap import base, creator, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt
from deap.tools.emo import sortNondominated
from data_parser import parse_data_file
from collections import defaultdict
import csv
import pandas as pd

# CONSTANTS
DEFAULT_SEED = 332
DEFAULT_POP_SIZE = 100
DEFAULT_CX_PROB = 0.2
DEFAULT_LAMBDA = 10
DEFAULT_MUT_PROB = 0.2
DEFAULT_N_GEN = 100

random.seed(DEFAULT_SEED)

##### LOAD AND PREPARE DATA #####
file_path = 'ptask (1)/data_1_23_40_66.dat'
parsed_data = parse_data_file(file_path)
num_workers = parsed_data['num_workers']
num_jobs = parsed_data['num_jobs']
qualifications = parsed_data['qualifications']
job_times = parsed_data['jobs']

#invert qualifications dict to get workers and their jobs
job_to_workers = defaultdict(list) 
for worker_id, qualified_jobs in qualifications.items():
    for job_id in qualified_jobs:
        job_to_workers[job_id].append(worker_id)

##### HELPER FUNCTIONS #####
# invalidate an individual if an unqualified worker is assigned to a job
def is_valid_assignment(individual, qualifications):
    """check if an individual has valid assignments"""
    for job_id, worker in enumerate(individual):
        if worker not in qualifications or job_id not in qualifications[worker]:
            return False
    return True

def calculate_worker_count(individual):
    """calculate the number of distinct workers used"""
    worker_count = len(set(individual))
    #print(f"Worker count: {worker_count} for individual: {individual}")
    return worker_count

def calculate_fairness(individual):
    """calculate the fairness of job assignments aka how evenly spread shifts are between workers"""
    from collections import Counter
    counts = Counter(individual)
    workloads = list(counts.values())
    mean = sum(workloads) / len(workloads)
    variance = sum((x - mean) ** 2 for x in workloads) / len(workloads)
    return -variance #neg so we can maximise fairness in the MOGA

def overlap_penalty(individual):
    from collections import defaultdict
    assigned = defaultdict(list)
    penalty = 0
    for job, w in enumerate(individual):
        assigned[w].append(job_times[job])
    for times in assigned.values():
        times.sort()
        for (s1, e1), (s2, e2) in zip(times, times[1:]):
            if s2 < e1:
                penalty +=1
    return penalty

def custom_mutation(individual, indpb):
    """Custom mutation function to ensure valid assignments"""
    for job_id in range(len(individual)):
        if random.random() < indpb:
            if job_id in job_to_workers and job_to_workers[job_id]:
                # Assign a random worker from the list of qualified workers
                individual[job_id] = random.choice(job_to_workers[job_id])
    return individual,

##### SET UP DEAP #####

# set up the problems fitness
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# initialisation of chromosomes
def create_individual():
    individual = []
    for job_id in range(num_jobs):
        if job_id in job_to_workers and job_to_workers[job_id]:
            # assign a random worker from the list of qualified workers
            individual.append(random.choice(job_to_workers[job_id]))
        else: 
            raise ValueError(f"No qualified workers for job {job_id}")
    return individual

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("select", tools.selNSGA2)

# register the evaluation functs
def evaluate(individual, lambda_penalty):
    if not is_valid_assignment(individual, qualifications):
        return float('inf'), 0 # worst possible fitness for if any worker is not qualified for their job
    worker_count = calculate_worker_count(individual)
    fairness_score = calculate_fairness(individual)
    overlap = overlap_penalty(individual)
    #print(f"evaluating individual: {individual} -> fitness: ({worker_count}, {qualified_worker_count})")
    return worker_count + lambda_penalty * overlap, fairness_score

def count_violations(individual, qualifications, job_times):
    qualification_errors = 0
    overlap_errors = 0
    assigned_jobs = {}
    for job_idx, worker in enumerate(individual):
        # qualification check
        if worker not in qualifications or job_idx not in qualifications[worker]:
            qualification_errors += 1
        # overlap check
        job_time = job_times[job_idx]
        if worker not in assigned_jobs:
            assigned_jobs[worker] = [job_time]
        else:
            for other_time in assigned_jobs[worker]:
                # check for overlap (start1 < end2) and (start2 < end1)
                if (job_time[0] < other_time[1]) and (other_time[0] < job_time[1]):
                    overlap_errors += 1
            assigned_jobs[worker].append(job_time)
    return qualification_errors, overlap_errors


##### MAIN LOOP #####

def is_valid(individual, qualifications, job_times):
    assigned_jobs = {}
    for job_idx, worker in enumerate(individual):
        # check qualification
        if worker not in qualifications[job_idx]:
            return False
        
        # check time overlap
        job_time = job_times[job_idx]
        if worker not in assigned_jobs:
            assigned_jobs[worker] = [job_time]
        else:
            for other_time in assigned_jobs[worker]:
                # check for overlap (start1 < end2) and (start2 < end1)
                if (job_time[0] < other_time[1]) and (other_time[0] < job_time[1]):
                    return False
            assigned_jobs[worker].append(job_time)
    return True
# create initial pop
def run_algorithm(pop_size, cx_prob, mut_prob, lambda_penalty, n_gen):
    toolbox.register("mutate", custom_mutation, indpb=mut_prob)
    toolbox.register("evaluate", lambda ind: evaluate(ind, lambda_penalty))
    population = toolbox.population(n=pop_size)

    for ind in population:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind)

    # initialise for visualisations
    pareto_history = []
    worker_count_history = []
    fairness_history = []
    # initialise the archive
    archive = []
    archive_history = []

    # set up alg params
    # apply crossover, mutation, and selection
    for gen in range(n_gen):
        offspring = list(map(toolbox.clone, population))

        # crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        # evaluate the new pop
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        else: 
            print("No invalid individuals to evaluate.")

        # select the next generation individuals
        population[:] = toolbox.select(population + offspring, len(population))
        combined = archive + population

        # get only non-dominated individuals
        archive = sortNondominated(combined, k=len(combined), first_front_only=True)[0]

        #track fitness valuyes and pareto front
        best_worker_count = min([ind.fitness.values[0] for ind in population]) # for minimisation
        worker_count_history.append(best_worker_count)

        best_fairness = max([ind.fitness.values[1] for ind in population]) # for maximisation
        fairness_history.append(best_fairness)
        
        archive_history.append([
            (ind.fitness.values[0], ind.fitness.values[1])
            for ind in archive if ind.fitness.valid])
        
        if gen in {0, int(n_gen/4), int(n_gen/2), int((2 * n_gen) / 4), n_gen-1}:
            pareto_front = sortNondominated(population, k=len(population), first_front_only=True)[0]
            pareto_history.append((gen, [(ind.fitness.values[0], ind.fitness.values[1]) for ind in pareto_front if ind.fitness.valid]))


    return worker_count_history, fairness_history, archive_history, pareto_history, archive

def is_valid(individual, qualifications, job_times):
    assigned_jobs = {}
    for job_idx, worker in enumerate(individual):
        # 1. Check qualification
        if worker not in qualifications[job_idx]:
            return False

        # 2. Check time overlap
        job_time = job_times[job_idx]
        if worker not in assigned_jobs:
            assigned_jobs[worker] = [job_time]
        else:
            for other_time in assigned_jobs[worker]:
                # Check for overlap: (start1 < end2) and (start2 < end1)
                if (job_time[0] < other_time[1]) and (other_time[0] < job_time[1]):
                    return False
            assigned_jobs[worker].append(job_time)

    return True


##### Plotting #####
def plot_results(worker_count_history, fairness_history, archive_history, pareto_history, archive,
                pop_size, cx_prob, mut_prob, lambda_penalty, n_gen):
    
    tag = f"pop{pop_size}_cx{cx_prob}_mut{mut_prob}_lambda{lambda_penalty}_gen{n_gen}"
    gens = list(range(1, len(worker_count_history)+1))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6,4))
    ax1.plot(gens, worker_count_history, label="Min Workers + Penalty")
    ax1.set_ylabel("Workers $+$ $\lambda \cdot$ Overlap")
    ax1.legend()

    ax2.plot(gens, fairness_history, label="Max Fairness (neg Var)", color="C1")
    ax2.set_ylabel("Fairness")
    ax2.set_xlabel("Generation")
    ax2.legend()

    plt.tight_layout()
    filename = f"convergence_{tag}.png"
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()

    # worker count plot
    plt.figure()
    plt.plot(worker_count_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Worker $+$ Penalty Count (minimise)")
    plt.title(f"Best Worker $+$ Penalty Count Over Generations \n({tag})", fontsize=10)
    plt.grid(True)
    filename = f"worker_count_{tag}.png"
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()

    # fairness plot
    plt.figure()
    plt.plot(fairness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fairness Score (maximise)")
    plt.title(f"Best Fairness Score Over Generations \n({tag})", fontsize=10)
    plt.grid(True)
    filename = f"fairness_history_{tag}.png"
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()

    # pareto front evolution plot
    plt.figure()
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'black']
    for (gen, front), color in zip(pareto_history, colors):
        xs = [pt[0] for pt in front]
        ys = [pt[1] for pt in front]
        plt.scatter(xs, ys, label=f"Gen {gen+1}", alpha=0.6, color=color)

    plt.xlabel('Number of Workers $+$ $\lambda \cdot$ Overlap (minimise)')
    plt.ylabel("Fairness Score (maximise)")
    plt.title(f"Pareto Evo \n({tag})", fontsize=10)
    plt.legend()
    plt.grid(True)
    filename = f"pareto_front_evolution_{tag}.png"
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()

    # final pareto front plot
    plt.figure()
    worker_counts = [ind.fitness.values[0] for ind in archive]
    fairness_score = [ind.fitness.values[1] for ind in archive]

    plt.scatter(worker_counts, fairness_score)
    plt.xlabel('Number of Workers $+$ $\lambda \cdot$ Overlap (minimise)')
    plt.ylabel('Fairness Score (maximise)')
    plt.title(f"Pareto Front \n({tag})", fontsize=10)
    plt.grid(True)
    filename = f"pareto_front_{tag}.png"
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()

##### ENTRY #####

if __name__ == "__main__":
    final_run_only = True #set to false to tune
    if final_run_only:
        print("Running final run with default parameters...")
        wh, fh, ah, ph, arch = run_algorithm(DEFAULT_POP_SIZE, DEFAULT_CX_PROB, DEFAULT_MUT_PROB, DEFAULT_LAMBDA, DEFAULT_N_GEN)
        plot_results(wh, fh, ah, ph, arch, DEFAULT_POP_SIZE, DEFAULT_CX_PROB, DEFAULT_MUT_PROB, DEFAULT_LAMBDA, DEFAULT_N_GEN)
        
        print("Checking validity of final Pareto front")

        results = []
        seen = set()
        for ind in arch:
            genome = tuple(ind)
            if genome in seen:
                continue
            seen.add(genome)

            qual_errors, overlaps = count_violations(ind, qualifications, job_times)
            workers = calculate_worker_count(ind)
            fairness = calculate_fairness(ind)

            #append to list
            results.append({
                "workers": workers,
                "fairness": fairness,
                "qual_errors": qual_errors,
                "overlaps": overlaps,
            })
        df = pd.DataFrame(results)
        df.to_csv("final_run_results.csv", index=False)
            
        print("Final run complete. Plots saved.")
    else:
        lambda_values = [5, 10, 20]
        mutation_probs = [0.1, 0.2, 0.3]
        crossover_probs = [0.2, 0.5, 0.7]
        pop_sizes = [50, 100, 200]
        n_gens = [50, 100, 200]

        key_plot_configs = [
            (50, 0.1, 10, 100),
            (100, 0.2, 10, 100),
            (200, 0.3, 10, 100),
            (100, 0.2, 5, 100),
            (100, 0.2, 20, 100)
        ]

        #prepare summary csv
        with open("summary_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Population Size", "Mutation Probability", "Lambda Penalty", "Best Worker Count", "Best Fairness Score"])

            for pop in pop_sizes:
                for mut in mutation_probs:
                    for lmb in lambda_values:
                        for ng in n_gens:
                            print(f"\nRunning with pop={pop}, mut_prob={mut}, lambda={lmb}, n_gen={ng}")
                            wh, fh, ah, ph, arch = run_algorithm(pop, DEFAULT_CX_PROB, mut, lmb, ng)

                            # selective plotting only for specific combos
                            if (pop, mut, lmb, ng) in key_plot_configs:
                                plot_results(wh, fh, ah, ph, arch, pop, DEFAULT_CX_PROB, mut, lmb, ng)
                            # Save the results to a file

                            # calc true best from final archive
                            best_ind = max(arch, key=lambda ind: calculate_fairness(ind))
                            bw = calculate_worker_count(best_ind)
                            bf = calculate_fairness(best_ind)
                            print(f"Best Worker Count: {bw}, Best Fairness Score: {bf:.3f}")
                            writer.writerow([pop, mut, lmb, ng, bw, f"{bf:.3f}"])

        print("All runs complete. Summary and plots saved.")