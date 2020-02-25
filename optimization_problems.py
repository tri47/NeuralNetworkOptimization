import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import mlrose
from random import randint

# PROBLEM 1 - Four Peak
def problem_1(state):
    global func_eval_count
    fitness = 0
    length = len(state)
    cutoff = round(length*0.15)
    start_ones = 0
    end_zeros = 0
    for i in state:
        if i == 0:
            break
        else:
            start_ones += 1
    for i in np.flip(state):
        if i == 1:
            break
        else:
            end_zeros += 1
    fitness = max(start_ones,end_zeros)
    if (start_ones > cutoff) & (end_zeros > cutoff):
        fitness += length
    func_eval_count += 1
    return fitness

# PROBLEM 2. Count 1 with all-or-nothing leading 1's
def problem_2(state):
    global func_eval_count
    fitness = 0
    state_length = len(state)
    state = np.array([state])
    ones = np.count_nonzero(state == 1)
    cutoff = round(state_length*0.1)
    cutoff = min(cutoff, 5)
    start_ones = np.count_nonzero(state[0,0:cutoff] == 1)

    fitness = fitness + ones
    if start_ones != cutoff:
        fitness -= start_ones
    func_eval_count += 1
    return fitness


# PROBLEM 3 - Snapsack
# Set up pre-defined weights for up to 200-bit string
weight_modifier = 0.5
weights = np.random.randint(1,10,200)
values = np.random.randint(1,10,200)
max_weight = weights[0:100].sum()*weight_modifier
def problem_3(state):
    global func_eval_count
    fitness = 0
    total_weight = 0
    max_weight = weights[0:len(state)].sum()*weight_modifier
    for i in range(0,len(state)):
        fitness += state[i]*values[i]
        total_weight += state[i]*weights[i]
        if total_weight > max_weight:
            fitness = 0 
            break
    func_eval_count += 1
    return fitness

def generate_results(fitness, init_state, max_val):
    global func_eval_count
    max_runs = 10
    length = np.shape(init_state)[0]
    print('Problem length: ', length)

    states = []
    for i in range(0,max_runs):
        new_state = np.random.randint(0,max_val,length)
        states.append(new_state)

    fitness_result = 0
    iter_result = 0 
    all_time = 0
    random_state = 100    
    func_eval_count = 0
    for i in range(max_runs):
        problem = mlrose.DiscreteOpt(length = length, fitness_fn = fitness, maximize = True, max_val = max_val)
        start_time = time.time()
        best_state, best_fitness, curve = mlrose.random_hill_climb(problem, 
                                                        max_attempts = 100, max_iters = 1000, restarts = 5,
                                                        init_state = states[i], random_state = random_state,curve = True)
        all_time += time.time() - start_time
        fitness_result += best_fitness
        iter_result += len(curve)

    rhc_curve = curve
    rhc_fitness = fitness_result/max_runs
    rhc_iters = iter_result/max_runs
    rhc_time = all_time/max_runs
    rhc_func_eval = func_eval_count/max_runs
    print('Random Hill Climb : ',rhc_fitness)
    print(rhc_iters)
    print('func evals: ', rhc_func_eval)
    

    fitness_result = 0
    iter_result = 0 
    all_time = 0
    random_state = 100    
    func_eval_count = 0
    for i in range(max_runs):
        problem = mlrose.DiscreteOpt(length = length, fitness_fn = fitness, maximize = True, max_val = max_val)
        start_time = time.time()
        best_state, best_fitness, curve = mlrose.simulated_annealing(problem, schedule = mlrose.ExpDecay(),
                                                        max_attempts = 100, max_iters = 1000, 
                                                        init_state = states[i], random_state = random_state,curve = True)
        all_time += time.time() - start_time
        fitness_result += best_fitness
        iter_result += len(curve)

    sa_curve = curve
    sa_fitness = fitness_result/max_runs
    sa_iters = iter_result/max_runs
    sa_time = all_time/max_runs
    sa_func_eval = func_eval_count/max_runs
    print('Simulated annealing: ',sa_fitness)
    print(sa_iters)
    print('func evals: ', sa_func_eval)
    
    
    fitness_result = 0
    iter_result = 0
    all_time = 0

    random_state = 100
    func_eval_count = 0
    for i in range(max_runs):
        problem = mlrose.DiscreteOpt(length = length, fitness_fn = fitness, maximize = True, max_val = max_val)
        start_time = time.time()
        best_state, best_fitness, curve = mlrose.genetic_alg(problem, 
                                                        max_attempts = 100, max_iters = 1000,pop_size=300, 
                                                        random_state = random_state, curve = True, mutation_prob = 0.1)
        all_time += time.time() - start_time
        fitness_result += best_fitness
        iter_result += len(curve)

    ga_curve = curve
    ga_fitness = fitness_result/max_runs
    ga_iters = iter_result/max_runs
    ga_time = all_time/max_runs
    ga_func_eval = func_eval_count/max_runs
    print('GA : ',ga_fitness)
    print(ga_iters)
    print('func evals: ', ga_func_eval)    

    fitness_result = 0
    iter_result = 0
    all_time = 0
    random_state = 100
    func_eval_count = 0
    for i in range(max_runs):
        print('MIMIC run: ', i)
        problem = mlrose.DiscreteOpt(length = length, fitness_fn = fitness, maximize = True, max_val = max_val)
        start_time = time.time()
        best_state, best_fitness, curve = mlrose.mimic(problem, 
                                                        max_attempts = 100, max_iters = 1000,  pop_size=300, keep_pct=0.2, 
                                                        random_state = random_state, curve = True)
        all_time += time.time() - start_time
        fitness_result += best_fitness
        iter_result += len(curve)
        print('fitness ', best_fitness)

    mi_curve = curve
    mi_fitness = fitness_result/max_runs
    mi_iters = iter_result/max_runs
    mi_time = all_time/max_runs
    mi_func_eval = func_eval_count/max_runs
    print('MIMIC: ',mi_fitness)
    print(mi_iters)
    print('func evals: ', mi_func_eval)

    return  rhc_fitness, rhc_func_eval, rhc_iters, rhc_curve, rhc_time, \
            sa_fitness, sa_func_eval, sa_iters, sa_curve, sa_time, \
            ga_fitness, ga_func_eval, ga_iters, ga_curve, ga_time, \
            mi_fitness, mi_func_eval, mi_iters, mi_curve, mi_time
            

  
def run_experiment(in_fitness_function, max_val):
    lengths = [10,20,30, 50, 100]
    ga_fitnesses = []
    sa_fitnesses = []
    mi_fitnesses = []
    rhc_fitnesses = []

    ga_func_evals, sa_func_evals, mi_func_evals, rhc_func_evals = [], [], [], []
    rhc_iters, ga_iters, sa_iters, mi_iters = [], [], [], []
    rhc_curves, sa_curves, ga_curves, mi_curves = [], [], [], []
    rhc_times, sa_times, ga_times, mi_times = [], [], [], []
    
    fitness = mlrose.CustomFitness(in_fitness_function)

    for length in lengths:
        #init_state = np.array([0]*length)
        init_state = np.random.randint(0,max_val,length)

        rhc_fitness, rhc_func_eval, rhc_iter, rhc_curve, rhc_time, \
            sa_fitness, sa_func_eval, sa_iter, sa_curve, sa_time, \
            ga_fitness, ga_func_eval, ga_iter, ga_curve, ga_time, \
            mi_fitness, mi_func_eval, mi_iter, mi_curve, mi_time = generate_results(fitness, init_state, max_val)
        rhc_fitnesses.append(rhc_fitness)
        sa_fitnesses.append(sa_fitness)
        ga_fitnesses.append(ga_fitness)
        mi_fitnesses.append(mi_fitness)
        rhc_iters.append(rhc_iter)
        sa_iters.append(sa_iter)
        ga_iters.append(ga_iter)
        mi_iters.append(mi_iter)
        rhc_times.append(rhc_time)
        sa_times.append(sa_time)
        ga_times.append(ga_time)
        mi_times.append(mi_time)
        rhc_func_evals.append(rhc_func_eval)
        sa_func_evals.append(sa_func_eval)
        ga_func_evals.append(ga_func_eval)
        mi_func_evals.append(mi_func_eval)



  #  best_fitness = [sa_fitnesses.max(), ga_fitnesses.max(), mi_fitnesses.max()]
    plt.figure(1)
    plt.plot(lengths, rhc_fitnesses)
    plt.plot(lengths, sa_fitnesses,'--+')
    plt.plot(lengths, ga_fitnesses, '--o')
    plt.plot(lengths, mi_fitnesses,'--')
    plt.legend(['Randomized Hill Climbing','Simulated Annealing', 'Genetic Algorithm', 'MIMIC'],fontsize=11)
    plt.title('Average fitness of each algorithm',fontsize=14)
    plt.xlabel('Problem length (bits)',fontsize=14)
    plt.ylabel('Average fitness',fontsize=14)
    plt.xticks(lengths)

    
    plt.figure(2)
    plt.plot(lengths, rhc_iters)
    plt.plot(lengths, sa_iters,'--+')
    plt.plot(lengths, ga_iters,'--o')
    plt.plot(lengths, mi_iters,'--')
    plt.legend(['Randomized Hill Climbing','Simulated Annealing', 'Genetic Algorithm', 'MIMIC'],fontsize=11)
    plt.title('Average iterations required',fontsize=14)
    plt.xlabel('Problem length (bits)',fontsize=14)
    plt.ylabel('Required iterations',fontsize=14)
    plt.xticks(lengths)

    plt.figure(3)
    plt.plot(rhc_curve)
    plt.plot(sa_curve, '--+')
    plt.plot(ga_curve, '--o')
    plt.plot(mi_curve,'--')
    plt.legend(['Randomized Hill Climbing','Simulated Annealing', 'Genetic Algorithm', 'MIMIC'],fontsize=11)
    plt.title('Fitness at each iternation',fontsize=14)
    plt.xlabel('Iteration',fontsize=14)
    plt.ylabel('Fitness',fontsize=14)

    plt.figure(4)
    plt.plot(lengths, rhc_times)
    plt.plot(lengths, sa_times,'--+')
    plt.plot(lengths, ga_times, '--o')
    plt.plot(lengths, mi_times,'--')
    plt.legend(['Randomized Hill Climbing','Simulated Annealing', 'Genetic Algorithm', 'MIMIC'],fontsize=11)
    plt.title('Average time required',fontsize=14)
    plt.xlabel('Problem length (bits)',fontsize=14)
    plt.ylabel('Time (s)',fontsize=14)
    plt.xticks(lengths)

    plt.figure(5)
    plt.plot(lengths, rhc_func_evals)
    plt.plot(lengths, sa_func_evals,'--+')
    plt.plot(lengths, ga_func_evals,'--o')
    plt.plot(lengths, mi_func_evals,'--')
    plt.legend(['Randomized Hill Climbing','Simulated Annealing', 'Genetic Algorithm', 'MIMIC'],fontsize=11)
    plt.title('Average function evaluations required',fontsize=14)
    plt.xlabel('Problem length (bits)',fontsize=14)
    plt.ylabel('Required function evaluations',fontsize=14)
    plt.xticks(lengths)
    plt.show()
    


if __name__ == "__main__" :
    import argparse
    print("Running Optimization Experiments")
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default='1')

    args = parser.parse_args()
    problem = args.problem
    func_eval_count = 0
    if problem == '1':
        print("Running Problem 1- Four Peak:...")
        run_experiment(problem_1,2)
    if problem == '2':
        print("Running Problem 2- Count 1 with all-or-nothing leading 1's:...")
        run_experiment(problem_2,2)
    if problem == '3':
        print("Running problem 3 - Snapsack:...")
        run_experiment(problem_3,2)



