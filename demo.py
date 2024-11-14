import os
import importlib
from tqdm import tqdm

from problem import Problem
from opti_functions import all_functions

algorithms_folder = 'algorithms'
results = {}
TIME_LIMIT = 10


print(f"Running all algorithms with TL of {TIME_LIMIT} seconds:")
print("-----------------")

for function_name, (fitness_func, LIMIT, EV) in all_functions.items():
    for python_file_name in tqdm(os.listdir(algorithms_folder), function_name, leave=False):
        if python_file_name.endswith('.py'):
            module_name = python_file_name[:-3]
            module = importlib.import_module(f'{algorithms_folder}.{module_name}')
            if hasattr(module, 'DIMENSION') and module.DIMENSION != len(LIMIT):
                continue
            importlib.reload(module)
            prob = Problem(fitness_func, LIMIT, EV, TIME_LIMIT)
            module.run(prob)
            results[module_name] = round(prob.best_solution[0], 5)

    print(f"Result for {function_name} function:", results)
    winner = [(name, value) for (name, value) in results.items() if value == min(results.values())][0]
    print(f"> Winner: {winner[0]} (expected: {EV}, got: {winner[1]}, error: {winner[1]-EV:.5f})", end="\n\n")
