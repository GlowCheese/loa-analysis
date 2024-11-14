import math

from math import pi
from typing import Callable

FitFunc = Callable[[list[float]], float]
all_functions: dict[str, tuple[FitFunc, float, float]] = {}

def _register(LIMIT: list[list[float]], EV: float):
    def _func_wrapper(fun):
        name = fun.__name__.split('_')[0]
        all_functions[name] = (fun, LIMIT, EV)
    return _func_wrapper


@_register([(-5.12, 5.12)] * 2, 0.0)
def rastrigin_function(pos: list[float]):
    d = len(pos)
    return 10*d + sum(x**2 - 10*math.cos(2*pi*x) for x in pos)


@_register([(-32.768, 32.768)] * 2, 0.0)
def ackley_function(pos: list[float]):
    a, b, c, d = 20, 0.2, 2*pi, len(pos)
    
    p1 = -a*math.exp(-b*math.sqrt(sum(x*x for x in pos)/d))
    p2 = -math.exp(sum(math.cos(c*x) for x in pos)/d)

    return p1 + p2 + a + math.exp(1)


@_register([(-15.0, -5.0), (-3.0, 3.0)], 0.0)
def bukin_function(pos: list[float]):
    x1, x2 = pos[0], pos[1]
    return 100*math.sqrt(abs(x2-0.01*x1*x1)) + 0.01*abs(x1+10)


@_register([(-5.12, 5.12)] * 2, -1.0)
def dropwave_function(pos: list[float]):
    x1, x2 = pos[0], pos[1]
    nume = 1+math.cos(12*math.sqrt(x1*x1+x2*x2))
    deno = 0.5*(x1*x1+x2*x2) + 2
    return -nume/deno