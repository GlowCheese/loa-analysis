import time
import random

from problem import Problem, INF


PRI_pop = 5000
NOM_pop = 1000
P = 100

MATE_rate = 0.3
SELE_rate = 0.6
IMMI_rate = 0.1
DEAD_rate = 0.3


nomads: list[tuple[list[float], float]] = []
prides: list[list[tuple[list[float], float]]] = [[] for _ in range(P)]

def run(prob: Problem):
    start_time = time.time()

    for _ in range(PRI_pop):
        pride = random.choice(prides)
        pos = prob.random_solution()
        pride.append([pos, prob.fitness(pos)])

    for _ in range(NOM_pop):
        pos = prob.random_solution()
        nomads.append([pos, prob.fitness(pos)])
    
    while int(time.time() - start_time) < prob.TIME_LIMIT:
        missing: list[int] = []

        for i, pride in enumerate(prides):
            org_size = len(pride)

            new_cubs_count = 0
            for _ in range(int(org_size * MATE_rate)):
                cnt = 0
                cub = [0] * prob.N_var
                for lion, _ in pride:
                    if random.random() > MATE_rate: continue
                    for i in range(prob.N_var):
                        cub[i] += lion[i]
                    cnt += 1
                if cnt == 0: continue

                new_cubs_count += 1
                cub = [x / cnt for x in cub]
                pride.append([cub, prob.fitness(cub)])

            pride.sort(key=lambda e: e[1])
            for _ in range(new_cubs_count + int(org_size * IMMI_rate)):
                nomads.append(pride.pop())

            missing += [i] * int(org_size * IMMI_rate)

        nomads.sort(key=lambda e: e[1], reverse=True)
        random.shuffle(missing)
        for i in missing:
            prides[i].append(nomads.pop())

        shifted = len(nomads) - NOM_pop
        for i in range(NOM_pop):
            if random.random() < DEAD_rate:
                pos = prob.random_solution()
                nomads[i] = [pos, prob.fitness(pos)]
            else:
                nomads[i] = nomads[i + shifted]

        for _ in range(shifted):
            nomads.pop()