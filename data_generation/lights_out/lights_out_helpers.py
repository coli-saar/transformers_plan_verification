import os
import random
from copy import copy
from collections import defaultdict
from planning_utils.validate_plan import run_validation


def create_grid(x, y, name_pref: str = 'button'):
    adjacencies = []
    all_locs = []

    # Create a mapping from (row, col) to location name
    def loc_name(row, col, name_prefix):
        return f"{name_prefix}_{row}_{col}"

    for row in range(y):
        for col in range(x):
            current = loc_name(row, col, name_pref)
            all_locs.append(current)
            # Up
            if row > 0:
                adjacencies.append((current, loc_name(row - 1, col, name_pref)))
            # Down
            if row < y - 1:
                adjacencies.append((current, loc_name(row + 1, col, name_pref)))
            # Left
            if col > 0:
                adjacencies.append((current, loc_name(row, col - 1, name_pref)))
            # Right
            if col < x - 1:
                adjacencies.append((current, loc_name(row, col + 1, name_pref)))

    return adjacencies, all_locs


def create_default_goal(all_locs):
    goal = []
    for loc in all_locs:
        goal.append(f'(light_out {loc})')
    return goal


def create_grid_init(grid, all_locs):
    init = []
    adj_dict = defaultdict(list)

    for pair in grid:
        init.append(f'(adjacent {pair[0]} {pair[1]})')
        adj_dict[pair[0]].append(pair[1])

    state_dict = dict()
    for loc in all_locs:
        state_dict[loc] = 'out'

    return init, adj_dict, state_dict


def execute_plan(current_state, adjacency_dict, plan):
    for action in plan:
        cleaned_action = action.replace('(', '').replace(')', '')
        target_button = cleaned_action.split(' ')[-1]
        assert target_button.startswith('button_')

        neighbors = adjacency_dict[target_button]

        buttons_to_change = neighbors + [target_button]
        for button in buttons_to_change:
            if current_state[button] == 'on':
                current_state[button] = 'out'
            elif current_state[button] == 'out':
                current_state[button] = 'on'
            else:
                raise ValueError

    return current_state


def get_incomplete_plan_conditional(plan_file_path,
                                    incorrect_file_path,
                                    problem_file_path,
                                    all_locs,
                                    domain_file,
                                    random_ind):
    with open(plan_file_path, 'r') as f:
        plan = []
        for line in f.readlines():
            if line.strip().startswith(';'):
                continue
            elif line.strip() == '(PLAN_END)':
                continue
            plan.append(line.strip())

    attempts = 0
    found_incorrect = False
    while attempts < 4:
        attempts += 1
        all_valid_actions = [f'(press_button {button})' for button in all_locs]
        if random_ind:
            ind = random.randint(0, len(plan) - 1)
        else:
            ind = -1

        # Replace original action with a random, applicable action
        orig_action = plan[ind]
        plan_incorrect = copy(plan)
        all_valid_actions.remove(orig_action)   # avoid sampling the original action
        new_action = random.choice(all_valid_actions)
        plan_incorrect[ind] = new_action

        with open(incorrect_file_path, 'w') as f:
            f.write('\n'.join(plan_incorrect))

        reached_goal, executable = run_validation(domain_file=domain_file,
                                                  problem_file=problem_file_path,
                                                  plan_file=incorrect_file_path,
                                                  plan_end_tag=True)

        if not reached_goal and executable:
            return plan_incorrect, ind

    if not found_incorrect:
        os.remove(incorrect_file_path)
        print(f'Did not find incorrect version for {plan_file_path}')
        raise ValueError

