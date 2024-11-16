import time
import random

from problem import Problem


N_pop = 100
MUTA_rate = 0.01


def roulette_wheel_selection(fitness_values):
    total_fitness = sum(1.0 / (1.0 + fitness) for fitness in fitness_values)
    random_value = random.random()
    cumulative_sum = 0

    for i, fitness in enumerate(fitness_values):
        cumulative_sum += (1.0 / (1.0 + fitness)) / total_fitness
        if cumulative_sum >= random_value:
            return i
    return len(fitness_values) - 1


def crossover(parent1, parent2, N_var):
    offspring = []
    k = random.randint(0, N_var - 1)
    r = random.random()
    for i in range(N_var):
        if i < k:
            offspring.append(parent1[i] * r + (1 - r) * parent2[i])
        else:
            offspring.append(parent2[i] * r + (1 - r) * parent1[i])
    return offspring


def mutate(individual, LIMIT, MUTA_rate):
    for i in range(len(individual)):
        if random.random() <= MUTA_rate:
            if(random.random() > 0.5 ) :
                individual[i] += random.uniform(LIMIT[i][0], LIMIT[i][1]) * 0.1
            else :
                individual[i] -= random.uniform(LIMIT[i][0], LIMIT[i][1]) * 0.1
            individual[i] = max(LIMIT[i][0], min(LIMIT[i][1], individual[i]))


def run(prob: Problem):
    population = [prob.random_solution() for _ in range(N_pop)]

    start_time = time.time()

    while int(time.time() - start_time) < prob.TIME_LIMIT:
        fitness_values = [prob.fitness(individual) for individual in population]
        new_population = []

        for _ in range(N_pop):
            parent1_index = roulette_wheel_selection(fitness_values)
            parent2_index = roulette_wheel_selection(fitness_values)

            offspring = crossover(population[parent1_index], population[parent2_index], prob.N_var)
            
            mutate(offspring, prob.LIMIT, MUTA_rate)
            
            new_population.append(offspring)

        population = new_population