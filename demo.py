import os
import importlib
from tqdm import tqdm
from colorama import Fore, Back, Style

from problem import Problem, INF
from opti_functions import all_functions

algorithms_folder = 'algorithms'
TIME_LIMIT = 10


file = open("output.txt", "w")
def dprint(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=file)

dprint()
dprint(f"   Running all algorithms with TL of {TIME_LIMIT} seconds:", end='\n\n')

max_function_name_len = max(len(function_name) for function_name in all_functions)
algorithms = []
for python_file_name in os.listdir(algorithms_folder):
    if python_file_name.endswith('.py'):
        module_name = python_file_name[:-3]
        module = importlib.import_module(f'{algorithms_folder}.{module_name}')
        algorithms.append((module_name, module))

dprint("   " + ' ' * max_function_name_len, '|', ' | '.join(f"{algo_name:^10}" for algo_name, _ in algorithms))

for function_name, (fitness_func, LIMIT, EV) in all_functions.items():
    results = []
    for algo_name, module in tqdm(algorithms, function_name, leave=False):
        if hasattr(module, 'DIMENSION') and module.DIMENSION != len(LIMIT):
            results.append((INF, 0))
        else:
            importlib.reload(module)
            prob = Problem(fitness_func, LIMIT, EV, TIME_LIMIT)
            module.run(prob)
            l = len(str(int(prob.best_solution[0])))
            results.append((round(prob.best_solution[0], 7-l), 7-l))

    _best = [output == min(output for output, _ in results) for output, _ in results]
    sbest = None if sum(_best) == len(results) else [
        output == min(output for (output, _), is_best
                      in zip(results, _best) if not is_best) for output, _ in results]

    dprint(f"   {function_name:>{max_function_name_len}}", end=' | ')
    dprint(
        ' | '.join(
        (Back.GREEN + Fore.WHITE if is_best
         else Back.YELLOW + Fore.BLACK if issbest and sum(_best) == 1 else '') +
        (f"{result:^10.{_l}f}" if result != INF else f"{'N/A':^10}") + Style.RESET_ALL
        for (result, _l), is_best, issbest in zip(results, _best, sbest))
    )

dprint()
file.close()