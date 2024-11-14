import os
import importlib

from problem import Problem
from opti_functions import all_functions

algorithms_folder = 'algorithms'
results = {}

print("Running each algorithm with TL of 3 seconds:")
print("-----------------")

for function_name, (fitness_func, LIMIT, EV) in all_functions.items():
    for python_file_name in os.listdir(algorithms_folder):
        if python_file_name.endswith('.py'):
            module_name = python_file_name[:-3]
            module = importlib.import_module(f'{algorithms_folder}.{module_name}')
            if hasattr(module, 'DIMENSION') and module.DIMENSION != len(LIMIT):
                continue
            prob = Problem(fitness_func, LIMIT, EV, 3)
            module.run(prob)
            results[module_name] = round(prob.best_solution[0], 4)

    print(f"Result for {function_name} function:", results)

print()