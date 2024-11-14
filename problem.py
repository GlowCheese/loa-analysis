import time
import random
import functools

from typing import Callable


INF = 999999999999


def _normalize_position(fun):
    @functools.wraps(fun)
    def f(self, pos: list[float]) -> float:
        pos = [
            max(
                self.LIMIT[i][0],
                min(self.LIMIT[i][1], pos[i])
            )
            for i in range(self.N_var)
        ]
        return fun(self, pos)
    return f


class Problem:
    def __init__(
        self,
        fitness_func: Callable[[list[float]], float],
        LIMIT: list[list[float]],
        EV: float,
        TIME_LIMIT: int = 10
    ):
        self.N_var = len(LIMIT)
        self.LIMIT = LIMIT
        self.fitness_func = fitness_func
        self.EV = EV
        self.TIME_LIMIT = TIME_LIMIT

        self.start_time = time.time()
        self.best_at_mili = {}
        self.best_solution = (INF, [0] * self.N_var)

        
    @_normalize_position
    def fitness(self, pos: list[float]) -> float:
        assert len(pos) == self.N_var
        res = self.fitness_func(pos)
        if res < self.best_solution[0]:
            end_time = time.time()
            at_mili = int(end_time - self.start_time)
            if at_mili <= self.TIME_LIMIT:
                self.best_solution = (res, pos)
                self.best_at_mili[at_mili] = res
        return res


    def random_coor(self, p: int) -> float:
        return random.uniform(self.LIMIT[p][0], self.LIMIT[p][1])


    def random_solution(self) -> list[float]:
        res = [self.random_coor(p) for p in range(self.N_var)]
        self.fitness(res)  # update solution if needed
        return res