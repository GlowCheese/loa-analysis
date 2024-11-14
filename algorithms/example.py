import time
import random

from problem import Problem


N_pop = 5000
MUTATION = 0.4


def run(prob: Problem):
    start_time = time.time()

    best_solution = prob.random_solution()
    best_fitness = prob.fitness(best_solution)

    population = [
        prob.random_solution()
        for _ in range(N_pop)
    ]
    
    while int(time.time() - start_time) < prob.TIME_LIMIT:
        for entity in population:
            for i in range(prob.N_var):
                if random.random() < MUTATION:
                    entity[i] = prob.random_coor(i)
            
            if (temp := prob.fitness(entity)) < best_fitness:
                best_fitness = temp
                best_solution = entity
