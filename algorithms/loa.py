import time
import math
import random

from enum import Enum
from math import pi, e

from problem import Problem, INF

N_pop = 3000
P = 50

NOM_rate = 0.4
FEM_rate = 0.7
HUNT_rate = 0.6
ROAM_rate = 0.5
MATE_rate = 0.4
MUTA_rate = 0.6
IMMI_rate = 0.4

prob: Problem = None


class Gender(Enum):
    MALE = 1
    FEMALE = 2


class Wing(Enum):
    LEFT = 1
    CENTER = 2
    RIGHT = 3


def PoI(old_fitness: float, new_fitness: float):
    diff = (old_fitness - new_fitness)  # always non-negative
    return diff / (abs(new_fitness) + 0.1)


class Lion:
    def __init__(self):
        self.position = prob.random_solution()
        self.improved_last_iter = True
        self.best_fitness = prob.fitness(self.position)

    def self_improve(self) -> float | None:
        new_fitness = prob.fitness(self.position)
        if new_fitness < self.best_fitness:
            res = PoI(self.best_fitness, new_fitness)
            self.improved_last_iter = True
            self.best_fitness = new_fitness
            return res
        else:
            self.improved_last_iter = False


prides: list[dict[Gender, list[Lion]]] = []
nomads: dict[Gender, list[Lion]] = {Gender.MALE: [], Gender.FEMALE: []}

nomad_size = lambda: len(nomads[Gender.MALE]) + len(nomads[Gender.FEMALE])
pride_size = lambda id: len(prides[id][Gender.MALE]) + len(prides[id][Gender.FEMALE])

def initialize_population():
    unalocated = []
    for _ in range(N_pop):
        if random.random() <= NOM_rate:
            if random.random() <= FEM_rate:
                nomads[Gender.MALE].append(Lion())
            else:
                nomads[Gender.FEMALE].append(Lion())
        else:
            unalocated.append(Lion())
    
    m = len(unalocated)
    for i in range(P):
        sz = (i+1)*m//P - i*m//P
        male_sz = max(2, int(random.normalvariate(sz*(1-FEM_rate), 1) + 0.5))
        prides.append({Gender.MALE: [], Gender.FEMALE: []})
        for _ in range(male_sz):
            prides[i][Gender.MALE].append(unalocated.pop())
        for _ in range(sz - male_sz):
            prides[i][Gender.FEMALE].append(unalocated.pop())


new_born = [0] * P

class Behaviour:
    @staticmethod
    def _move_toward(lion: Lion, dest: list[float], angle: float = pi/2 - pi/1024):
        buff: float = random.uniform(0, 2)
        for p in range(prob.N_var):
            x: float = buff * (dest[p] - lion.position[p])
            lion.position[p] += x * math.tan(random.uniform(-angle, angle))


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
                        min(prey[p], 2 * prey[p] - lion.position[p]),
                        max(prey[p], 2 * prey[p] - lion.position[p])
                    )

            improvement_percentage = lion.self_improve()
            if improvement_percentage is not None:
                intensity: float = random.uniform(0, improvement_percentage)
                for p in range(prob.N_var):
                    prey[p] += intensity * (prey[p] - lion.position[p])


    @staticmethod
    def _fleeing(pride_id: int, fleers: list[Lion]):
        success_count = sum(1 for lion in fleers if lion.improved_last_iter)
        tournament_size = max(2, (success_count + 1) // 2)

        females = prides[pride_id][Gender.FEMALE]
        males = prides[pride_id][Gender.MALE]
        _fem_size: int = len(females)

        for lion in fleers:
            target: Lion = None
            for _ in range(tournament_size):
                r = random.randint(0, len(females) + len(males) - 1)
                chosen = females[r] if r < _fem_size else males[r - _fem_size]

                if target is None or chosen.best_fitness < target.best_fitness:
                    target = chosen

            for p in range(prob.N_var):
                Behaviour._move_toward(lion, target.position)
                lion.self_improve()


    @staticmethod
    def hunting_or_fleeing():
        for i in range(P):
            fleers: list[Lion] = []
            hunters: list[tuple[Lion, str]] = []
            prey: list[float] = [0] * prob.N_var

            for lion in prides[i][Gender.FEMALE]:
                if random.random() <= HUNT_rate:
                    wing: str = random.choice(list(Wing))
                    hunters.append((lion, wing))
                    for p in range(prob.N_var):
                        prey[p] += lion.position[p]
                else:
                    fleers.append(lion)

            if hunters:
                prey = [p / len(hunters) for p in prey]

            Behaviour._hunting(prey, hunters)
            Behaviour._fleeing(i, fleers)


    @staticmethod
    def roaming():
        for i in range(P):
            _fem_size = len(prides[i][Gender.FEMALE])
            for lion in prides[i][Gender.MALE]:
                for _ in range(int(pride_size(i) * ROAM_rate)):
                    r = random.randint(0, pride_size(i) - 1)
                    if r < _fem_size:
                        Behaviour._move_toward(lion, prides[i][Gender.FEMALE][r].position, pi/6)
                    else:
                        Behaviour._move_toward(lion, prides[i][Gender.MALE][r - _fem_size].position, pi/6)
                    lion.self_improve()

        best_nomad = INF
        for lion in nomads[Gender.FEMALE] + nomads[Gender.MALE]:
            best_nomad = min(best_nomad, prob.fitness(lion.position))

        for lion in nomads[Gender.FEMALE] + nomads[Gender.MALE]:
            pr = 0.1 + min(0.5, PoI(prob.fitness(lion.position), best_nomad))
            if random.random() > pr:
                lion.position = prob.random_solution()
                lion.self_improve()


    @staticmethod
    def _breed(female: Lion, males: list[Lion], pride: dict[Gender, list[Lion]]):
        beta = random.gauss(0.5, 0.1)
        off_1 = Lion()
        off_2 = Lion()

        for p in range(prob.N_var):
            sum_positions = sum(male.position[p] for male in males)
            if random.random() < MUTA_rate:
                off_1.position[p] = prob.random_coor(p)
            else:
                off_1.position[p] = beta * female.position[p] + (1 - beta) * sum_positions / len(males)
            
            if random.random() < MUTA_rate:
                off_2.position[p] = prob.random_coor(p)
            else:
                off_2.position[p] = (1 - beta) * female.position[p] + beta * sum_positions / len(males)

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
            if random.random() > MATE_rate:
                continue
            assert nomads[Gender.MALE], "No males available"
            female = nomads[Gender.FEMALE][j]
            r = random.randint(0, len(nomads[Gender.MALE]) - 1)
            males = [nomads[Gender.MALE][r]]
            Behaviour._breed(female, males, nomads)


    @staticmethod
    def defense():
        for i in range(P):
            prides[i][Gender.MALE].sort(key=lambda u: u.best_fitness)
            while new_born[i] > 0:
                nomads[Gender.MALE].append(prides[i][Gender.MALE].pop())
                new_born[i] -= 1

        for k in range(len(nomads[Gender.MALE])):
            ok = False
            for i in range(P):
                if random.random() > 0.5:
                    continue
                for j in range(len(prides[i][Gender.MALE]) - 1, -1, -1):
                    if nomads[Gender.MALE][k].best_fitness < prides[i][Gender.MALE][j].best_fitness:
                        nomads[Gender.MALE].append(prides[i][Gender.MALE][j])
                        prides[i][Gender.MALE].append(nomads[Gender.MALE][k])
                        nomads[Gender.MALE].pop(k)
                        prides[i][Gender.MALE].pop(j)
                        ok = True
                        break
                if ok: break


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
        nomads[Gender.FEMALE].sort(key=lambda u: u.best_fitness, reverse=True)

        for i in missing:
            assert nomads[Gender.FEMALE], "No females available"
            prides[i][Gender.FEMALE].append(nomads[Gender.FEMALE].pop())


    @staticmethod
    def equilibrium():
        lim = {Gender.MALE: int(N_pop * NOM_rate * FEM_rate), Gender.FEMALE: int(N_pop * NOM_rate * (1 - FEM_rate))}

        for gender in [Gender.MALE, Gender.FEMALE]:
            nomads[gender].sort(key=lambda u: u.best_fitness)
            if len(nomads[gender]) > lim[gender]:
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

