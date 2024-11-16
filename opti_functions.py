import math

from typing import Callable
from math import pi, sin, cos, tanh, exp, sqrt, prod

FitFunc = Callable[[list[float]], float]
all_functions: dict[str, tuple[FitFunc, float, float]] = {}

def _register(_LIMIT: list[list[float]], _EV: float):
    def _func_wrapper(fun):
        name = fun.__name__
        EV = 1.0*_EV
        LIMIT = [(1.0*mn, 1.0*mx) for (mn, mx) in _LIMIT]
        all_functions[name] = (fun, LIMIT, EV)
    return _func_wrapper


@_register([(-5.12, 5.12)] * 2, 0.0)
def rastrigin(pos: list[float]):
    d = len(pos)
    return 10*d + sum(x**2 - 10*cos(2*pi*x) for x in pos)


@_register([(-32.768, 32.768)] * 2, 0.0)
def ackley(pos: list[float]):
    a, b, c, d = 20, 0.2, 2*pi, len(pos)
    
    p1 = -a*exp(-b*sqrt(sum(x*x for x in pos)/d))
    p2 = -exp(sum(cos(c*x) for x in pos)/d)

    return p1 + p2 + a + exp(1)


@_register([(-15.0, -5.0), (-3.0, 3.0)], 0.0)
def bukin(pos: list[float]):
    x1, x2 = pos[0], pos[1]
    return 100*sqrt(abs(x2-0.01*x1*x1)) + 0.01*abs(x1+10)


@_register([(-5.12, 5.12)] * 2, -1.0)
def drop_wave(pos: list[float]):
    x1, x2 = pos[0], pos[1]
    nume = 1+cos(12*sqrt(x1*x1+x2*x2))
    deno = 0.5*(x1*x1+x2*x2) + 2
    return -nume/deno


@_register([(-10, 10)] * 2, 0.0)
def himmelblau(pos: list[float]):
    x, y = pos[0], pos[1]
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


@_register([(-512, 512)] * 2, -959.6407)
def eggholder(pos: list[float]):
    x, y = pos[0], pos[1]
    term1 = -(y + 47) * sin(sqrt(abs(x/2 + (y + 47))))
    term2 = -x * sin(sqrt(abs(x - (y + 47))))
    return term1 + term2


@_register([(-10, 10)] * 2, -2.06261187)
def cross_in_tray(pos: list[float]):
    x, y = pos[0], pos[1]
    term1 = sin(x) * sin(y)
    term2 = exp(abs(100 - sqrt(x**2 + y**2) / pi))
    return -0.0001 * (abs(term1 * term2) + 1)**0.1


@_register([(-10, 10)] * 2, 0.0)
def levi(pos: list[float]):
    x, y = pos[0], pos[1]
    term1 = sin(3 * pi * x)**2
    term2 = (x - 1)**2 * (1 + sin(3 * pi * y)**2)
    term3 = (y - 1)**2 * (1 + sin(2 * pi * y)**2)
    return term1 + term2 + term3


@_register([(-100, 100)] * 2, 0.0)
def griewank(pos: list[float]):
    sum_term = sum(x**2 / 4000 for x in pos)
    prod_term = prod(cos(x / sqrt(i + 1)) for i, x in enumerate(pos))
    return sum_term - prod_term + 1


@_register([(-100, 100)] * 2, 0.0)
def schaffer(pos: list[float]):
    x, y = pos[0], pos[1]
    nume = sin(x**2 - y**2)**2 - 0.5
    deno = (1 + 0.001 * (x**2 + y**2))**2
    return 0.5 + nume / deno


@_register([(0, pi)] * 5, -4.687658)
def michalewicz(pos: list[float]):
    m = 10
    return -sum(sin(x) * sin((i + 1) * x**2 / pi)**(2 * m) for i, x in enumerate(pos))


@_register([(-500, 500)] * 2, 0.0)
def schwefel(pos: list[float]):
    d = len(pos)
    return 418.9829 * d - sum(x * sin(sqrt(abs(x))) for x in pos)


@_register([(1, 60)] * 5, 0.0)
def villiers_glasser(pos: list[float]):
    x1, x2, x3, x4, x5 = pos
    result = 0
    for i in range(1, 25):
        t = 0.1*(i-1)
        y = 53.81*(1.27**t)*tanh(3.012*t+sin(2.13*t))*cos(exp(0.507)*t)
        result += x1*x2**t*tanh(x3*t+sin(x4*t))*cos(t*exp(x5)-y)**2
    return result


@_register([(-10, 10)] * 2, -186.7309)
def shubert(pos: list[float]):
    x, y = pos[0], pos[1]
    
    def single_dim_shubert(z):
        return sum(i * cos((i + 1) * z + i) for i in range(1, 6))
    
    return single_dim_shubert(x) * single_dim_shubert(y)


@_register([(-4, 4)] * 2, -2*(2+4)*(2-1)/6)
def trid(pos: list[float]):
    n = len(pos)
    sum1 = sum((pos[i] - 1) ** 2 for i in range(n))
    sum2 = sum(pos[i] * pos[i-1] for i in range(1, n))
    return sum1 - sum2


@_register([(-5.12, 5.12)] * 2, 0.0)
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


@_register([(0, 14)] * 2, 0.0)
def damavandi(pos: list[float]):
    x1, x2 = pos
    p1 = sin(pi*(x1-2))*sin(pi*(x2-2))
    p2 = pi*pi*(x1-2)*(x2-2)
    p3 = (1-abs(p1/p2)**5)
    p4 = 2+(x1-7)**2+2*(x2-7)**2
    return p3 + p4


@_register([(-10, 10)] * 2, -1.0)
def cross_leg_table(pos: list[float]):
    x1, x2 = pos
    p1 = abs(100-sqrt(x1**2+x2**2)/pi)
    p2 = abs(sin(x1)*sin(x2)*exp(p1))
    return -1/(p2+1)**0.1


@_register([(-10.24, 10.24)] * 2, 0.0)
def whitley(pos: list[float]):
    return sum(
        p1**2/4000-cos(p1)+1
        for xi in pos for xj in pos
        for p1 in [100*(xi**2-xj)**2+(1-xj)**2]
    )


@_register([(-10, 10)] * 2, 0.0001)
def crowned_cross(pos: list[float]):
    x1, x2 = pos
    p1 = 100 - sqrt(x1**2+x2**2)/pi
    p2 = exp(abs(p1))*sin(x1)*sin(x2)
    return 0.0001*(abs(p2)+1)**0.1


@_register([(0, 4)] + [(-4, 4)] * 16, 11.7464)
def cola(pos: list[float]):
    u = [0] + pos
    x = [0, 0, u[1]] + [u[2*(i-2)] for i in range(3, 11)]
    y = [0, 0, 0] + [u[2*(i-2)+1] for i in range(3, 11)]
    d = [[], [],
         [0, 1.27],
         [0, 1.69, 1.43],
         [0, 2.04, 2.35, 2.43],
         [0, 3.09, 3.18, 3.26, 2.85],
         [0, 3.20, 3.22, 3.27, 2.88, 1.55],
         [0, 2.86, 2.56, 2.58, 2.59, 3.12, 3.06],
         [0, 3.17, 3.18, 3.18, 3.12, 1.31, 1.64, 3.00],
         [0, 3.21, 3.18, 3.18, 3.17, 1.70, 1.36, 2.95, 1.32],
         [0, 2.38, 2.31, 2.42, 1.94, 2.85, 2.81, 2.56, 2.91, 2.97]
         ]
    return sum(
        (r-d[i][j])**2
        for i in range(2, 11) for j in range(1, i)
        for r in [sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)]
    )