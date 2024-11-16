import time
import math
import random

from math import pi
from enum import Enum

from problem import Problem, INF

N_pop = 2000
P = 70

NOM_rate = 0.3
FEM_rate = 0.7
HUNT_rate = 0.7
ROAM_rate = 0.7
MATE_rate = 0.2
MUTA_rate = 0.4
IMMI_rate = 0.15
DISS_rate = 0.02

prob: Problem = None


class Gender(Enum):
    MALE = 1
    FEMALE = 2


class Wing(Enum):
    LEFT = 1
    CENTER = 2
    RIGHT = 3


def PoI(old_fitness: float, new_fitness: float):
    diff = (old_fitness - new_fitness)
    return diff / (abs(new_fitness) + 0.1)


class Lion:
    def __init__(self):
        self.position = prob.random_solution()
        self.improved_last_iter = True
        self.cr_fitness = prob.fitness(self.position)

    def self_improve(self) -> float | None:
        new_fitness = prob.fitness(self.position)
        res = max(0, PoI(self.cr_fitness, new_fitness))
        self.improved_last_iter = new_fitness < self.cr_fitness
        self.cr_fitness = new_fitness
        return res

    def __str__(self):
        return f"{self.cr_fitness} at position({', '.join(map(str, self.position))})"


prides: list[dict[Gender, list[Lion]]] = []
nomads: dict[Gender, list[Lion]] = {Gender.MALE: [], Gender.FEMALE: []}

nomad_size = lambda: len(nomads[Gender.MALE]) + len(nomads[Gender.FEMALE])
pride_size = lambda id: len(prides[id][Gender.MALE]) + len(prides[id][Gender.FEMALE])


def initialize_population():
    unalocated = []
    num_nomads = N_pop * NOM_rate

    for i in range(N_pop):
        if i >= num_nomads:
            unalocated.append(Lion())
        elif random.random() <= FEM_rate:
            nomads[Gender.MALE].append(Lion())
        else:
            nomads[Gender.FEMALE].append(Lion())
    
    m = len(unalocated)
    for i in range(P):
        sz = (i+1)*m//P - i*m//P
        male_sz = max(2, int(random.gauss(sz*(1-FEM_rate), 1)))

        prides.append({Gender.MALE: [], Gender.FEMALE: []})
        for _ in range(male_sz):
            prides[i][Gender.MALE].append(unalocated.pop())
        for _ in range(sz - male_sz):
            prides[i][Gender.FEMALE].append(unalocated.pop())


new_born = [0] * P

class Behaviour:
    @staticmethod
    def _move_toward(lion: Lion, dest: list[float], angle: float = pi/2 - pi/1024):
        buff = random.uniform(0, 2)
        for p in range(prob.N_var):
            x = math.tan(random.uniform(-angle, angle))
            lion.position[p] += buff * (dest[p] - lion.position[p] + x)
        lion.self_improve()


    @staticmethod
    def _hunting(prey: list[float], hunters: list[tuple[Lion, Wing]]):
        for lion, wing in hunters:
            for p in range(prob.N_var):
                if wing == Wing.CENTER:
                    lion.position[p] = random.uniform(
                        min(lion.position[p], prey[p]),
                        max(lion.position[p], prey[p])
                    )
                else:
                    lion.position[p] = random.uniform(
                        min(prey[p], 2*prey[p] - lion.position[p]),
                        max(prey[p], 2*prey[p] - lion.position[p])
                    )

            improvement_percentage = lion.self_improve()
            for p in range(prob.N_var):
                intensity = random.uniform(0, improvement_percentage)
                prey[p] += intensity * (lion.position[p] - prey[p])


    @staticmethod
    def _fleeing(pride_id: int, fleers: list[Lion]):
        success_count = \
            sum(lion.improved_last_iter for lion in prides[pride_id][Gender.FEMALE]) \
            + sum(lion.improved_last_iter for lion in prides[pride_id][Gender.MALE])
        tournament_size = max(2, (success_count + 1) // 2)

        males = prides[pride_id][Gender.MALE]
        females = prides[pride_id][Gender.FEMALE]
        _fem_size = len(females)

        for lion in fleers:
            target: Lion = None
            for _ in range(tournament_size):
                r = random.randrange(pride_size(pride_id))
                chosen = females[r] if r < _fem_size else males[r - _fem_size]

                if target is None or chosen.cr_fitness < target.cr_fitness:
                    target = chosen

            for p in range(prob.N_var):
                Behaviour._move_toward(lion, target.position)


    @staticmethod
    def generate_prey(hunters: list[tuple[Lion, str]]) -> list[float]:
        mn = min(lion[0].cr_fitness for lion in hunters)
        mx = max(lion[0].cr_fitness for lion in hunters)

        if mn == mx: e = [1] * len(hunters)
        else: e = [1-(lion[0].cr_fitness-mn)/(mx-mn) for lion in hunters]
        
        s = sum(e)
        return [
            sum(e[i]*hunters[i][0].position[p] for i in range(len(hunters))) / s
            for p in range(prob.N_var)
        ]


    @staticmethod
    def hunting_or_fleeing():
        for i in range(P):
            fleers: list[Lion] = []
            hunters: list[tuple[Lion, str]] = []

            for lion in prides[i][Gender.FEMALE]:
                if random.random() <= HUNT_rate:
                    wing = random.choice(list(Wing))
                    hunters.append((lion, wing))
                else:
                    fleers.append(lion)

            if len(hunters) >= 2:
                prey = Behaviour.generate_prey(hunters)
                Behaviour._hunting(prey, hunters)

            if fleers:
                Behaviour._fleeing(i, fleers)


    @staticmethod
    def roaming():
        for i in range(P):
            _fem_size = len(prides[i][Gender.FEMALE])
            for lion in prides[i][Gender.MALE]:
                for _ in range(int(pride_size(i) * ROAM_rate)):
                    r = random.randrange(pride_size(i))
                    if r < _fem_size:
                        Behaviour._move_toward(lion, prides[i][Gender.FEMALE][r].position, pi/6)
                    else:
                        Behaviour._move_toward(lion, prides[i][Gender.MALE][r - _fem_size].position, pi/6)

        best_nomad = min(
            min(lion.cr_fitness for lion in nomads[Gender.MALE]),
            min(lion.cr_fitness for lion in nomads[Gender.FEMALE])
        )

        for lion in nomads[Gender.FEMALE] + nomads[Gender.MALE]:
            pr = 0.1 + min(0.5, PoI(lion.cr_fitness, best_nomad))
            if random.random() < pr:
                lion.position = prob.random_solution()
                lion.self_improve()


    @staticmethod
    def _breed(female: Lion, males: list[Lion], pride: dict[Gender, list[Lion]]):
        beta = random.gauss(0.5, 0.1)
        off_1, off_2 = Lion(), Lion()

        for p in range(prob.N_var):
            sum_positions = sum(male.position[p] for male in males)
            off_1.position[p] = beta * female.position[p] + (1 - beta) * sum_positions / len(males)
            off_2.position[p] = (1 - beta) * female.position[p] + beta * sum_positions / len(males)
            if random.random() < MUTA_rate: off_1.position[p] += math.tan(random.uniform(-pi/2, pi/2))
            if random.random() < MUTA_rate: off_2.position[p] += math.tan(random.uniform(-pi/2, pi/2))

        off_1.self_improve()
        off_2.self_improve()

        if random.random() < 0.5:
            pride[Gender.FEMALE].append(off_1)
            pride[Gender.MALE].append(off_2)
        else:
            pride[Gender.MALE].append(off_1)
            pride[Gender.FEMALE].append(off_2)


    @staticmethod
    def mating():
        for i in range(P):
            for jf in range(len(prides[i][Gender.FEMALE]) - 1, -1, -1):
                if random.random() > MATE_rate:
                    continue

                assert prides[i][Gender.MALE], "No males available"
                female = prides[i][Gender.FEMALE][jf]

                males = []
                while not males:
                    for jm in range(len(prides[i][Gender.MALE])):
                        if random.random() < 1 - MATE_rate:
                            males.append(prides[i][Gender.MALE][jm])

                new_born[i] += 1
                Behaviour._breed(female, males, prides[i])

        for j in range(len(nomads[Gender.FEMALE]) - 1, -1, -1):
            if random.random() > MATE_rate: continue
            assert nomads[Gender.MALE], "No males available"
            female = nomads[Gender.FEMALE][j]
            r = random.randrange(len(nomads[Gender.MALE]))
            males = [nomads[Gender.MALE][r]]
            Behaviour._breed(female, males, nomads)


    @staticmethod
    def defense():
        for i in range(P):
            prides[i][Gender.MALE].sort(key=lambda u: u.cr_fitness)
            while new_born[i] > 0:
                nomads[Gender.MALE].append(prides[i][Gender.MALE].pop())
                new_born[i] -= 1

        for k in range(len(nomads[Gender.MALE])):
            for i in range(P):
                if random.random() > 0.5: continue
                u = len(prides[i][Gender.MALE]) - 1
                if nomads[Gender.MALE][k].cr_fitness < prides[i][Gender.MALE][u].cr_fitness:
                    prides[i][Gender.MALE].append(nomads[Gender.MALE][k])
                    nomads[Gender.MALE][k] = prides[i][Gender.MALE].pop(u)
                    while u > 0 and prides[i][Gender.MALE][u].cr_fitness < prides[i][Gender.MALE][u - 1].cr_fitness:
                        temp = prides[i][Gender.MALE][u]
                        prides[i][Gender.MALE][u] = prides[i][Gender.MALE][u-1]
                        prides[i][Gender.MALE][u-1] = temp
                        u -= 1
                    break


    @staticmethod
    def migration():
        missing: list[int] = []
        for i in range(P):
            _max_fem = int(len(prides[i][Gender.MALE]) / (1 - FEM_rate) * FEM_rate)
            rate = (max(0, len(prides[i][Gender.FEMALE]) - _max_fem) + IMMI_rate * _max_fem) / len(prides[i][Gender.FEMALE])
            for j in range(len(prides[i][Gender.FEMALE]) - 1, -1, -1):
                if random.random() < rate:
                    nomads[Gender.FEMALE].append(prides[i][Gender.FEMALE].pop(j))
            for _ in range(_max_fem - len(prides[i][Gender.FEMALE])):
                missing.append(i)

        random.shuffle(missing)
        nomads[Gender.FEMALE].sort(key=lambda u: u.cr_fitness, reverse=True)

        for i in missing:
            assert nomads[Gender.FEMALE], "No females available"
            prides[i][Gender.FEMALE].append(nomads[Gender.FEMALE].pop())


    @staticmethod
    def equilibrium():
        lim = {
            Gender.MALE: int(N_pop * NOM_rate * FEM_rate),
            Gender.FEMALE: int(N_pop * NOM_rate * (1 - FEM_rate))
        }
        for gender in list(Gender):
            if len(nomads[gender]) > lim[gender]:
                random.shuffle(nomads[gender])
                nomads[gender] = nomads[gender][:lim[gender]]


def run(_prob: Problem):
    global prob
    prob = _prob

    start_time = time.time()
    initialize_population()

    while int(time.time() - start_time) < prob.TIME_LIMIT:
        Behaviour.hunting_or_fleeing()
        Behaviour.roaming()
        Behaviour.mating()
        Behaviour.defense()
        Behaviour.migration()
        Behaviour.equilibrium()
