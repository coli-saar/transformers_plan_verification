import os
import json
import random
from copy import deepcopy
from pathlib import Path
from planning_utils.validate_plan import run_validation
from data_generation.lights_out.lights_out_helpers import create_grid, create_default_goal, create_grid_init, \
    execute_plan, get_incomplete_plan_conditional


def generate_dataset_conditional(data_dir,
                                 next_inst_id,
                                 n_per_len,
                                 domain_file,
                                 output_file,
                                 max_len,
                                 min_len,
                                 x_dim=5,
                                 y_dim=5,
                                 random_ind_incorrect: bool = True,
                                 rem_individual_files: bool = True
                                 ):
    problem_dir = os.path.join(data_dir, 'problems_pddl')
    plan_dir = os.path.join(data_dir, 'plans_pddl')
    if domain_file is None:
        domain_file = os.path.join(data_dir, 'domain.pddl')

    incorrect_plan_dir = os.path.join(data_dir, 'plans_incorrect')

    json_out_file = os.path.join(data_dir, output_file)

    Path(problem_dir).mkdir(exist_ok=True, parents=True)
    Path(plan_dir).mkdir(exist_ok=True, parents=True)
    Path(incorrect_plan_dir).mkdir(exist_ok=True, parents=True)
    dataset = []

    adjacencies, all_locs = create_grid(x=x_dim, y=y_dim)
    goal = create_default_goal(all_locs=all_locs)
    shared_init, adj_dict, state_dict = create_grid_init(adjacencies, all_locs)

    # do everything that is identical for different instances
    locs_str = ' '.join(all_locs)
    objects_str = f'\t(:objects\n\t\t{locs_str}\n)\n\n'

    goal_str = f'\t(:goal\n\t\t(and\n'
    for fact in goal:
        goal_str += f'\t\t\t{fact}\n'
    goal_str += f'\t\t)\n\t)'

    max_attempts = 4 * n_per_len

    for plan_len in range(min_len, max_len + 1):
        tasks_so_far = []
        generated = 0
        attempts = 0
        while generated < n_per_len:
            if attempts >= max_attempts:
                break
            print(f'Generating plans of length {plan_len}')
            init, plan_str = generate_from_empty(plan_len=plan_len,
                                                 all_locs=all_locs,
                                                 current_state=state_dict,
                                                 adj_dict=adj_dict)

            if (init, plan_str) in tasks_so_far:
                continue
            else:
                tasks_so_far.append((init, plan_str))

            init_str = '\t(:init\n'
            for fact in shared_init:
                init_str += f'\t\t{fact}\n'
            for fact in init:
                init_str += f'\t\t{fact}\n'
            init_str += '\t)\n\n'

            header_str = f'(define (problem light-inst-{next_inst_id})\n\t(:domain lights)\n\n'
            problem_str = f'{header_str}{objects_str}{init_str}{goal_str})'

            instance_path = os.path.join(problem_dir, f'instance-{next_inst_id}.pddl')
            plan_path = os.path.join(plan_dir, f'instance-{next_inst_id}_plan.txt')
            incorrect_plan_path = os.path.join(incorrect_plan_dir, f'instance-{next_inst_id}_incorrect.txt')

            with open(instance_path, 'w') as f:
                f.write(problem_str)

            with open(plan_path, 'w') as f:
                f.write(plan_str)

            reached_goal, executable = run_validation(domain_file=domain_file,
                                                      problem_file=instance_path,
                                                      plan_file=plan_path)
            assert reached_goal and executable

            incorrect_plan, _ = get_incomplete_plan_conditional(
                plan_file_path=plan_path,
                incorrect_file_path=incorrect_plan_path,
                problem_file_path=instance_path,
                domain_file=domain_file,
                random_ind=random_ind_incorrect,
                all_locs=all_locs
            )

            correct_data_inst = {
                'problem': problem_str,
                'plan': plan_str,
                'plan_length': plan_len,
                'n_obj': x_dim * y_dim,
                'incorrect_action_index': None
            }

            incorrect_data_inst = {
                'problem': problem_str,
                'plan': '\n'.join(incorrect_plan),
                'plan_length': len(incorrect_plan),
                'n_obj': x_dim * y_dim,
                'incorrect_action_index': -1
            }
            dataset.append(correct_data_inst)
            dataset.append(incorrect_data_inst)

            next_inst_id += 1
            generated += 1
            attempts += 1

            with open(json_out_file, 'a') as f:
                json.dump(correct_data_inst, f)
                f.write('\n')
                json.dump(incorrect_data_inst, f)
                f.write('\n')

            if rem_individual_files:
                os.remove(instance_path)
                os.remove(plan_path)
                os.remove(incorrect_plan_path)


def generate_from_empty(plan_len, all_locs, current_state, adj_dict):
    new_init_state = deepcopy(current_state)

    all_valid_actions = [f'(press_button {button})' for button in all_locs]
    plan = []
    while len(plan) < plan_len:
        action = random.choice(all_valid_actions)
        plan.append(action)

    # Execute to get new initial state
    new_init_state = execute_plan(current_state=new_init_state,
                                  adjacency_dict=adj_dict,
                                  plan=plan)

    init_facts = []
    for button, state in new_init_state.items():
        if state == 'on':
            init_facts.append(f'(light_on {button})')
        elif state == 'out':
            init_facts.append(f'(light_out {button})')
        else:
            raise ValueError

    init_facts.sort()

    plan.reverse()
    plan_str = '\n'.join(plan)

    return init_facts, plan_str
