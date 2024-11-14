import time
import math
import random
from copy import deepcopy

from problem import Problem


DIMENSION = 2
INF = 999999999999

num_ants = 100
rho = 0.05
alpha = 1.0
betaa = 2.0
A = 10
tau_min = 0.1
tau_max = 10.0
step_size = 0.3
Degrees = 360

pheromones = [tau_max] * Degrees


class Ant:
    def __init__(self):
        self.result = INF
        self.x = 0
        self.y = 0
        self.degree = 0
    
    def pos(self) -> list[float]:
        return [self.x, self.y]


MaxIter = Ant()
best_answer = Ant()



def update_pheromone(ants, MaxIter):
    global pheromones
    for i in range(Degrees):
        pheromones[i] *= (1.0 - rho)
    for ant in ants:
        if ant.result == MaxIter:
            pheromones[ant.degree] += rho * tau_max
        else:
            pheromones[ant.degree] += rho * tau_min


def selectDegree(ant, EV):
    probabilities = [0.0] * Degrees
    sum_probabilities = 0.0

    for i in range(Degrees):
        heuristic = 1.0 / (1.0 + ant.result - EV)
        probabilities[i] = (pheromones[i] ** alpha) * (heuristic ** betaa)
        sum_probabilities += probabilities[i]

    r = random.random()
    cumulative_probability = 0.0

    for i in range(Degrees):
        if probabilities[i] > 0:
            cumulative_probability += probabilities[i] / sum_probabilities
            if r <= cumulative_probability:
                return i
    return Degrees - 1


def run(prob: Problem):
    start_time = time.time()

    global MaxIter, best_answer

    ants = [Ant() for _ in range(num_ants)]
    best_answer.result = prob.fitness(best_answer.pos())
    
    while int(time.time() - start_time) < prob.TIME_LIMIT:
        position = prob.random_solution()
        MaxIter = Ant()

        ants[0].x = position[0]
        ants[0].y = position[1]
        ants[0].result = prob.fitness(position)

        for i in range(1, num_ants):
            dre = selectDegree(ants[i - 1], prob.EV)

            ants[i].x += math.cos(dre * math.pi / 180.0) * step_size
            ants[i].y += math.sin(dre * math.pi / 180.0) * step_size
            ants[i].result = prob.fitness(ants[i].pos())

            if ants[i].result < MaxIter.result:
                MaxIter = deepcopy(ants[i])

            if best_answer.result > MaxIter.result:
                best_answer = deepcopy(MaxIter)

        update_pheromone(ants, MaxIter)    
