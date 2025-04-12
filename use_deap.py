import re
import os
import random
from deap import base, creator, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt
from deap.tools.emo import sortNondominated

archive = []
archive_history = []
def parse_data_file(file_path):
    data = {}

    # Extract metadata from filename
    filename = os.path.basename(file_path)
    match = re.match(r'data_(\d+)_(\d+)_(\d+)_(\d+)\.dat', filename)
    if match:
        data['file_number'] = int(match.group(1))
        data['num_workers'] = int(match.group(2))
        data['num_jobs'] = int(match.group(3))  # double-check against file content
        data['multi_skilling_level'] = int(match.group(4))
    else:
        raise ValueError(f"Filename '{filename}' doesn't match expected pattern.")

    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    # Skip first 3 lines (comments)
    lines = lines[3:]

    # Line 2: Type
    type_match = re.search(r'Type\s*=\s*(\d+)', lines[0])
    if type_match:
        data['type'] = int(type_match.group(1))
    else:
        raise ValueError("Couldn't parse 'Type =' line.")

    # Next num_jobs lines: Job start and end times
    job_lines = lines[2:2 + data['num_jobs']]
    data['jobs'] = [tuple(map(int, line.split())) for line in job_lines]

    # Number of shifts (qualifications)
    shift_line_idx = 2 + data['num_jobs']
    num_qualifications = int(lines[shift_line_idx].split('=')[1].strip())
    data['num_qualifications'] = num_qualifications

    # Next lines: qualifications
    qualification_lines = lines[shift_line_idx + 1: shift_line_idx + 1 + num_qualifications]

    qualifications = {}
    for shift_idx, line in enumerate(qualification_lines):
        parts = line.replace(":", "").split()
        job_count = int(parts[0])  # Not strictly needed
        qualified_job_indices = list(map(int, parts[1:]))
        qualifications[shift_idx] = qualified_job_indices

    data['qualifications'] = qualifications

    return data


file_path = 'ptask (1)/data_1_23_40_66.dat'
parsed_data = parse_data_file(file_path)
print(parsed_data)
breakpoint()
num_workers = parsed_data['num_workers']
num_jobs = parsed_data['num_jobs']
qualifications = parsed_data['qualifications']
job_times = parsed_data['jobs']


def calculate_worker_count(individual):
    """calculate the number of distinct workers used"""
    worker_count = len(set(individual))
    #print(f"Worker count: {worker_count} for individual: {individual}")
    return worker_count,

def check_for_overlaps(individual, job_times):
    """check for overlaps in job assignments"""
    from collections import defaultdict

    worker_jobs = defaultdict(list)

    # assign hobs to workers based on the ind sol
    for job_id, worker in enumerate(individual):
        worker_jobs[worker].append(job_id)

        penalty = 0
        for jobs in worker_jobs.values():
            # sort jobs by their start time
            sorted_jobs = sorted(jobs, key = lambda job: job_times[job][0])

            # check for overlaps
            for i in range(1, len(sorted_jobs)):
                prev_job = sorted_jobs[i - 1]
                curr_job = sorted_jobs[i]
                prev_end = job_times[prev_job][1]
                curr_start = job_times[curr_job][0]
                if curr_start < prev_end:
                    penalty += 1
    return penalty

def calculate_qualified_worker_count(individual):
    """calculate the number of jobs where assigned worker is qualified"""
    qualified_jobs = 0
    for job, worker in enumerate(individual):
        if job in qualifications:
            if worker in qualifications[job]:
                qualified_jobs += 1
    #print(f"Qualified worker count: {qualified_jobs} for individual: {individual}")
    return qualified_jobs,

# set up the problems fitness
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# initialisation of chromosomes
def create_individual():
    return [random.randint(0, num_workers-1) for _ in range(num_jobs)]

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# register the evaluation functs
def evaluate(individual):
    worker_count = calculate_worker_count(individual)[0]
    qualified_worker_count = calculate_qualified_worker_count(individual)[0]
    overlap_penalty = check_for_overlaps(individual, job_times)
    adjusted_worker_count = worker_count + (10 * overlap_penalty)
    #print(f"evaluating individual: {individual} -> fitness: ({worker_count}, {qualified_worker_count})")
    return adjusted_worker_count, qualified_worker_count

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=num_workers-1, indpb=0.2)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluate)

# create initial pop
population = toolbox.population(n=100)

for ind in population:
    if not ind.fitness.valid:
        ind.fitness.values = toolbox.evaluate(ind)

pareto_history = []
# set up alg params
# apply crossover, mutation, and selection
for gen in range(100):
    print(f"Generation {gen + 1}")
    offspring = list(map(toolbox.clone, population))

    # crossover and mutation
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.7:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
            # reevaluate the children
            child1.fitness.values = toolbox.evaluate(child1)
            child2.fitness.values = toolbox.evaluate(child2)

    for mutant in offspring:
        toolbox.mutate(mutant)
        del mutant.fitness.values
            # reevaluate the mutant
        mutant.fitness.values = toolbox.evaluate(mutant)

    # evaluate the new pop
    for ind in population:
        print(f"individual: {ind} fitness before: {ind.fitness.values}")
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    if invalid_ind:
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        #print(f"fitnesses: {fitnesses}, invalid_ind: {invalid_ind}")
        for ind, fit in zip(invalid_ind, fitnesses):
            #print(f"individual: {ind} fitness before: {ind.fitness.values}")
            ind.fitness.values = fit
            #print(f"individual: {ind} fitness after: {ind.fitness.values}")
    else: 
        print("No invalid individuals to evaluate.")
    

    # select the next generation individuals
    population[:] = toolbox.select(population + offspring, len(population))
    combined = archive + population

    # get only non-dominated individuals
    archive = sortNondominated(combined, k=len(combined), first_front_only=True)[0]
    
    archive_history.append([
        (ind.fitness.values[0], ind.fitness.values[1])
        for ind in archive if ind.fitness.valid])
    if gen in {0, 10, 25, 50, 75, 99}:
        pareto_front = sortNondominated(population, k=len(population), first_front_only=True)[0]
        pareto_history.append((gen, [(ind.fitness.values[0], ind.fitness.values[1]) for ind in pareto_front if ind.fitness.valid]))
for ind in population:
    print(f"individual: {ind}, fitness: {ind.fitness.values}")

colors = ['red', 'orange', 'green', 'blue', 'purple', 'black']

#for (gen, front), color in zip(pareto_history, colors):
 #   xs = [pt[0] for pt in front]
  #  ys = [pt[1] for pt in front]
   # plt.scatter(xs, ys, label=f"Gen {gen+1}", alpha=0.6, color=color)

#plt.xlabel("Worker count (minimise)")
#plt.ylabel("Qualified worker count (maximise)")
#plt.title("Pareto Front Over Generations")
#plt.legend()
#plt.grid(True)
#plt.show()
worker_counts = [ind.fitness.values[0] for ind in archive]
qualified_worker_counts = [ind.fitness.values[1] for ind in archive]

plt.scatter(worker_counts, qualified_worker_counts)
plt.xlabel('Number of Workers')
plt.ylabel('Number of Qualified Workers')
plt.title('Pareto Front')
plt.grid(True)
plt.show()
