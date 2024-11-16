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
Degrees = 361

pheromones = [tau_max] * Degrees

prob = None


class Ant:
    def __init__(self):
        self.result = INF
        self.x = prob.random_coor(0)
        self.y = prob.random_coor(1)
        self.degree = 0
    
    def pos(self) -> list[float]:
        return [self.x, self.y]


MaxIter: Ant = None
best_answer: Ant = None



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

    position = prob.random_solution()

    for i in range(Degrees):
        position[0] = ant.x + math.cos(i * math.pi / 180.0) * step_size
        position[1] = ant.y + math.sin(i * math.pi / 180.0) * step_size

        heuristic = 1.0 / (1.0 + prob.fitness(position) - EV)
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


def run(_prob: Problem):
    global prob, MaxIter, best_answer

    prob = _prob
    start_time = time.time()
    best_answer = Ant()

    ants = [Ant() for _ in range(num_ants)]
    best_answer.result = prob.fitness(best_answer.pos())
    
    position = prob.random_solution()
    MaxIter = Ant()

    MaxIter.x = position[0]
    MaxIter.y = position[1]
    MaxIter.result = prob.fitness(position)

    while int(time.time() - start_time) < prob.TIME_LIMIT:
        ants[0] = deepcopy(MaxIter)

        for i in range(1, num_ants):
            dre = selectDegree(ants[i - 1], prob.EV)

            ants[i].x = ants[i-1].x + math.cos(dre * math.pi / 180.0) * step_size
            ants[i].y = ants[i-1].y + math.sin(dre * math.pi / 180.0) * step_size
            ants[i].result = prob.fitness(ants[i].pos())

            if ants[i].result < MaxIter.result:
                MaxIter = deepcopy(ants[i])

            if best_answer.result > MaxIter.result:
                best_answer = deepcopy(MaxIter)

        update_pheromone(ants, MaxIter)    
