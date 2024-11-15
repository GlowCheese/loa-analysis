import math

from math import pi
from typing import Callable

FitFunc = Callable[[list[float]], float]
all_functions: dict[str, tuple[FitFunc, float, float]] = {}

def _register(_LIMIT: list[list[float]], _EV: float):
    def _func_wrapper(fun):
        name = fun.__name__
        EV = 1.0*_EV
        LIMIT = [(1.0*mn, 1.0*mx) for (mn, mx) in _LIMIT]
        all_functions[name] = (fun, LIMIT, EV)
    return _func_wrapper


@_register([(-5.12, 5.12)] * 3, 0.0)
def rastrigin(pos: list[float]):
    d = len(pos)
    return 10*d + sum(x**2 - 10*math.cos(2*pi*x) for x in pos)


@_register([(-32.768, 32.768)] * 4, 0.0)
def ackley(pos: list[float]):
    a, b, c, d = 20, 0.2, 2*pi, len(pos)
    
    p1 = -a*math.exp(-b*math.sqrt(sum(x*x for x in pos)/d))
    p2 = -math.exp(sum(math.cos(c*x) for x in pos)/d)

    return p1 + p2 + a + math.exp(1)


@_register([(-15.0, -5.0), (-3.0, 3.0)], 0.0)
def bukin(pos: list[float]):
    x1, x2 = pos[0], pos[1]
    return 100*math.sqrt(abs(x2-0.01*x1*x1)) + 0.01*abs(x1+10)


@_register([(-5.12, 5.12)] * 2, -1.0)
def drop_wave(pos: list[float]):
    x1, x2 = pos[0], pos[1]
    nume = 1+math.cos(12*math.sqrt(x1*x1+x2*x2))
    deno = 0.5*(x1*x1+x2*x2) + 2
    return -nume/deno


@_register([(-10, 10)] * 2, 0.0)
def himmelblau(pos: list[float]):
    x, y = pos[0], pos[1]
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


@_register([(-512, 512)] * 2, -959.6407)
def eggholder(pos: list[float]):
    x, y = pos[0], pos[1]
    term1 = -(y + 47) * math.sin(math.sqrt(abs(x/2 + (y + 47))))
    term2 = -x * math.sin(math.sqrt(abs(x - (y + 47))))
    return term1 + term2


@_register([(-10, 10)] * 2, -2.06261)
def cross_in_tray(pos: list[float]):
    x, y = pos[0], pos[1]
    term1 = math.sin(x) * math.sin(y)
    term2 = math.exp(abs(100 - math.sqrt(x**2 + y**2) / math.pi))
    return -0.0001 * (abs(term1 * term2) + 1)**0.1


@_register([(-10, 10)] * 2, 0.0)
def levi(pos: list[float]):
    x, y = pos[0], pos[1]
    term1 = math.sin(3 * math.pi * x)**2
    term2 = (x - 1)**2 * (1 + math.sin(3 * math.pi * y)**2)
    term3 = (y - 1)**2 * (1 + math.sin(2 * math.pi * y)**2)
    return term1 + term2 + term3


@_register([(-5, 5)] * 2, 0.0)
def three_hump_camel(pos: list[float]):
    x, y = pos[0], pos[1]
    return 2*x**2 - 1.05*x**4 + (x**6) / 6 + x*y + y**2


@_register([(-100, 100)] * 2, 0.0)
def griewank(pos: list[float]):
    sum_term = sum(x**2 / 4000 for x in pos)
    prod_term = math.prod(math.cos(x / math.sqrt(i + 1)) for i, x in enumerate(pos))
    return sum_term - prod_term + 1


@_register([(-100, 100)] * 2, 0.0)
def schaffer(pos: list[float]):
    x, y = pos[0], pos[1]
    nume = math.sin(x**2 - y**2)**2 - 0.5
    deno = (1 + 0.001 * (x**2 + y**2))**2
    return 0.5 + nume / deno


@_register([(0, pi)] * 5, -4.687658)
def michalewicz(pos: list[float]):
    m = 10
    return -sum(math.sin(x) * math.sin((i + 1) * x**2 / math.pi)**(2 * m) for i, x in enumerate(pos))


@_register([(-500, 500)] * 3, 0.0)
def schwefel(pos: list[float]):
    d = len(pos)
    return 418.9829 * d - sum(x * math.sin(math.sqrt(abs(x))) for x in pos)


@_register([(1, 60)] * 5, 0.0)
def de_villiers_glasser_02(pos: list[float]):
    x1, x2, x3, x4, x5 = pos
    result = 0
    for i in range(1, 25):
        t = 0.1*(i-1)
        y = 53.81*(1.27**t)*math.tanh(3.012*t+math.sin(2.13*t))*math.cos(math.exp(0.507)*t)
        result += x1*x2**t*math.tanh(x3*t+math.sin(x4*t))*math.cos(t*math.exp(x5)-y)**2
    return result


@_register([(-10, 10)] * 2, -186.7309)
def shubert(pos: list[float]):
    x, y = pos[0], pos[1]
    
    def single_dim_shubert(z):
        return sum(i * math.cos((i + 1) * z + i) for i in range(1, 6))
    
    return single_dim_shubert(x) * single_dim_shubert(y)


@_register([(-49, 49)] * 7, -7*(7+4)*(7-1)/6)
def trid(pos: list[float]):
    n = len(pos)
    sum1 = sum((pos[i] - 1) ** 2 for i in range(n))
    sum2 = sum(pos[i] * pos[i-1] for i in range(1, n))
    return sum1 - sum2


@_register([(-5.12, 5.12)] * 6, 0.0)
def sphere(pos: list[float]):
    return sum(x ** 2 for x in pos)


@_register([(-10, 10), (-10, 10)], 0.0)
def matyas(pos: list[float]):
    x, y = pos[0], pos[1]
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


@_register([(0, 4)] * 4, 0.0)
def power_sum(pos: list[float]):
    b = [8, 18, 44, 114]
    return sum((sum(pos[j]**i for j in range(4)) - b[i-1])**2 for i in range(1, 5))


@_register([(-5, 10)] * 5, 0.0)
def rosenbrock(pos: list[float]):
    return sum(100*(pos[i+1]-pos[i]**2)**2+(pos[i]-1)**2 for i in range(4))
