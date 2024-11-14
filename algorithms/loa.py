import time

from problem import Problem

N_pop = 3000
P = 50

NOM_rate = 0.4
FEM_rate = 0.7
HUNT_rate = 0.6
ROAM_rate = 0.5
MATE_rate = 0.4
MUTA_rate = 0.6
IMMI_rate = 0.4

def run(prob: Problem):
    start_time = time.time()
    while int(time.time() - start_time) < prob.TIME_LIMIT:
        prob.random_solution()
