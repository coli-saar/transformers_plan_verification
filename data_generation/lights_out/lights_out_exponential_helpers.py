import os
import random
from itertools import product
import pymimir as mm
from collections import defaultdict
from planning_utils.execute_plan import execute_plan_mimir
from planning_utils.validate_plan import run_validation
from data_generation.lights_out.lights_out_helpers import create_grid_init, create_grid


def generate_domain_file_exp(x_dim, y_dim, out_file=None, write=True):

    if write:
        assert out_file

    adjacencies, all_locs = create_grid(x=x_dim, y=y_dim)
    all_locs.sort()
    locs_str = ' '.join(all_locs)

    _, adj_dict, _ = create_grid_init(grid=adjacencies, all_locs=all_locs)

    all_truth_combs5 = list(product([False, True], repeat=5))
    all_truth_combs4 = list(product([False, True], repeat=4))
    all_truth_combs3 = list(product([False, True], repeat=3))

    actions = dict()
    action_prec_and_effects = dict()
    for location, neighbors in adj_dict.items():

        if len(neighbors) == 4:
            required_truth_combs = all_truth_combs5
        elif len(neighbors) == 3:
            required_truth_combs = all_truth_combs4
        elif len(neighbors) == 2:
            required_truth_combs = all_truth_combs3
        else:
            raise ValueError

        neighbors.sort()

        for truth_comb in required_truth_combs:
            locs = [location] + neighbors
            assignment = []
            preconditions = []
            effects = []
            name = 'press'
            for l_id, loc in enumerate(locs):
                loc_name = loc.replace('button_', '')
                loc_name = loc_name.replace('_', '')

                truth_val = truth_comb[l_id]
                assignment.append((loc, truth_val))

                if truth_val:
                    loc_name += 't'
                    preconditions.append(f'(light_on {loc})')
                    preconditions.append(f'(not (light_out {loc}))')
                    effects.append(f'(light_out {loc})')
                    effects.append(f'(not (light_on {loc}))')
                else:
                    loc_name += 'f'
                    preconditions.append(f'(light_out {loc})')
                    preconditions.append(f'(not (light_on {loc}))')
                    effects.append(f'(light_on {loc})')
                    effects.append(f'(not (light_out {loc}))')
                name += f'_{loc_name}'

            actions[name] = assignment
            action_prec_and_effects[name] = {'prec': preconditions, 'effects': effects}

    content = f'(define (domain lights)\n\t(:requirements :strips :negative-preconditions)\n\n\t(:constants {locs_str})\n\n\t(:predicates\n\t\t(light_on ?b)\n\t\t(light_out ?b)\n\t\t(adjacent ?b1 ?b2)\n\t)\n\n'

    for action, action_scheme in action_prec_and_effects.items():
        action_str = f'\t(:action {action}\n\t\t:parameters ()\n'
        precondition_str = f'\t\t:precondition (and\n'
        for precond in action_scheme['prec']:
            precondition_str += f'\t\t\t{precond}\n'
        precondition_str += f')\n'
        effect_str = f'\t\t:effect (and\n'
        for effect in action_scheme['effects']:
            effect_str += f'\t\t\t{effect}\n'
        effect_str += f')\n\t)'

        action_str += f'{precondition_str}{effect_str}\n\n'
        content += action_str

    content += ')'

    if write:
        with open(out_file, 'w') as f:
            f.write(content)

    return actions, action_prec_and_effects


def create_empty_initial_state_file(all_locs,
                                    header_str,
                                    goal_str,
                                    objects_str,
                                    out_file,
                                    shared_init):

    init = []
    all_locs.sort()
    for button in all_locs:
        init.append(f'(light_out {button})')

    init_str = '\t(:init\n'
    for fact in shared_init:
        init_str += f'\t\t{fact}\n'
    for fact in init:
        init_str += f'\t\t{fact}\n'
    init_str += '\t)\n\n'

    problem_str = f'{header_str}{objects_str}{init_str}{goal_str})'

    with open(out_file, 'w') as f:
        f.write(problem_str)


def get_anonymized_action_names(actions):
    action_names = list(actions.keys())
    action_names.sort()
    action_mappings = dict()
    next_action_id = 0

    for ac in action_names:
        new_action_name = f'press_{next_action_id}'
        action_mappings[ac] = new_action_name
        next_action_id += 1

    return action_mappings


def get_mappings_semi_anonymized(actions):
    action_names = list(actions.keys())
    action_names.sort()
    action_mappings = dict()

    count_per_button = defaultdict(int)

    for ac in action_names:
        cleaned_ac_name = ac.replace('press_', '')
        button_to_press = cleaned_ac_name.split('_')[0]
        button_to_press = button_to_press.replace('t', '').replace('f', '')
        assert len(button_to_press) == 2
        button_to_press = f'{button_to_press[0]}_{button_to_press[1]}'

        action_id = count_per_button[button_to_press]
        count_per_button[button_to_press] += 1
        new_action_name = f'press_{button_to_press}_{action_id}'
        action_mappings[ac] = new_action_name

    return action_mappings


def anonymize_plan(action_mappings,
                   plan_str):

    new_plan = []
    plan = plan_str.split('\n')
    for ac in plan:
        ac_no_brackets = ac.replace('(', '').replace(')', '')
        new_ac = action_mappings[ac_no_brackets]
        new_plan.append(f'({new_ac})')
    new_plan_str = '\n'.join(new_plan)
    return new_plan_str


def generate_correct_plan_from_empty(domain_file_path,
                                     empty_instance_file_path,
                                     plan_len: int):
    """
    Also needs to generate the initial state!
    :param domain_file_path:
    :param empty_instance_file_path:
    :param plan_len:
    :return:
    """
    correct_plan_reversed = []

    parser = mm.PDDLParser(str(domain_file_path), str(empty_instance_file_path))
    problem = parser.get_problem()
    factories = parser.get_pddl_factories()

    successor_generator = mm.LiftedApplicableActionGenerator(problem, factories)
    state_repository = mm.StateRepository(successor_generator)
    current_state = state_repository.get_or_create_initial_state()

    all_applicable_actions_mimir = successor_generator.compute_applicable_actions(current_state)
    all_appl_actions_str = [action.to_string_for_plan(factories) for action in all_applicable_actions_mimir]
    action_mappings = {ac_str: ac_mm for (ac_str, ac_mm) in zip(all_appl_actions_str, all_applicable_actions_mimir)}

    while len(correct_plan_reversed) < plan_len:
        next_action = random.choice(all_appl_actions_str)
        correct_plan_reversed.append(next_action)

        current_state = state_repository.get_or_create_successor_state(
            current_state, action_mappings[next_action]
        )

        all_applicable_actions_mimir = successor_generator.compute_applicable_actions(current_state)
        all_appl_actions_str = [action.to_string_for_plan(factories) for action in all_applicable_actions_mimir]
        action_mappings = {ac_str: ac_mm for (ac_str, ac_mm) in zip(all_appl_actions_str, all_applicable_actions_mimir)}

    return correct_plan_reversed


def generate_init_state_and_plan(reversed_plan, all_locs, action_preconds_and_effects):
    init_state = []
    correct_plan = []
    all_locs.sort()
    for button in all_locs:
        init_state.append(f'(light_out {button})')

    for action in reversed_plan:
        cleaned_action = action[1:-1]   # get rid of brackets

        # get action for reverse direction
        reversed_action = cleaned_action.replace('f', 'F')
        reversed_action = reversed_action.replace('t', 'T')
        reversed_action = reversed_action.replace('F', 't')
        reversed_action = reversed_action.replace('T', 'f')
        correct_plan.append(f'({reversed_action})')

        effects = action_preconds_and_effects[cleaned_action]['effects']
        for effect in effects:
            if effect.startswith('(not'):
                delete_effect = effect[1:-1]    # get rid of brackets
                delete_effect = delete_effect.replace('not ', '')
                init_state.remove(delete_effect)
            else:
                init_state.append(effect)

    init_state.sort()
    correct_plan.reverse()

    return init_state, correct_plan


def generate_incomplete_plan(problem_file,
                             correct_plan_file,
                             incorrect_plan_file,
                             domain_file
                             ):
    correct_plan = []
    with open(correct_plan_file, 'r') as f:
        for line in f.readlines():
            action = line.strip()
            if action.startswith(';'):
                continue
            if action == '(PLAN_END)':
                continue
            correct_plan.append(action)

    if len(correct_plan) < 2:
        print(f'Not possible for {correct_plan_file}')
        return

    attempts = 0
    found_incorrect = False
    while attempts < 4:

        plan_incorrect = correct_plan[:-1]
        current_state, all_appl_actions_str = execute_plan_mimir(
            plan=plan_incorrect,
            domain_file=domain_file,
            problem_file=problem_file
        )
        all_appl_actions_str = [ac for ac in all_appl_actions_str if ac != correct_plan[-1]]
        new_action = random.choice(all_appl_actions_str)
        plan_incorrect.append(new_action)

        with open(incorrect_plan_file, 'w') as f:
            f.write('\n'.join(plan_incorrect))

        reached_goal, executable = run_validation(domain_file=domain_file,
                                                  problem_file=problem_file,
                                                  plan_file=incorrect_plan_file,
                                                  plan_end_tag=False)

        if not reached_goal and executable:
            return plan_incorrect, -1

    if not found_incorrect:
        os.remove(incorrect_plan_file)
        print(f'Did not find incorrect version for {correct_plan_file}')
        raise ValueError
