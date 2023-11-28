import numpy as np


def calc_choosen_utility(consumption, floor, disutil):
    unemployed_util, work_util = calc_utilities(consumption, disutil, floor)
    utilities = np.column_stack((unemployed_util, work_util))
    choice = determine_optimal_choice(utilities)
    utility_of_choice = np.take(utilities, choice)
    return utility_of_choice


def determine_optimal_choice(utilities):
    shocks = np.random.gumbel(size=(utilities.shape[0], 2))
    choice_specific_util = utilities + shocks
    return np.argmax(choice_specific_util, axis=1, keepdims=True)


def calc_utilities(cons, disutil, floor):
    floor_consumption = calc_floor_consumption(cons, floor)
    base_utility = np.log(floor_consumption)
    utility_work = base_utility - disutil
    return base_utility, utility_work


def calc_floor_consumption(cons, floor):
    mask = cons < floor
    cons[mask] = floor
    return cons


np.random.seed(1234)
max_grid = 1_000_000
num_grid = 1_000_000
consumption_agents = np.random.uniform(1, max_grid, num_grid)
floor_consumption = 0.1
disutility = 0.5

utility_of_choice = calc_choosen_utility(consumption_agents, floor_consumption, disutility)

np.save("consumption.npy", utility_of_choice)
