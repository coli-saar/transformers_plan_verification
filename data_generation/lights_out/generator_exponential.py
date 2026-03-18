import os
import json
import shutil
from pathlib import Path
from planning_utils.paths import TEMP_DIR
from data_generation.lights_out.lights_out_helpers import create_default_goal
from data_generation.lights_out.lights_out_exponential_helpers import *


def generate_dataset_exponential(data_dir,
                                 next_inst_id,
                                 n_per_len,
                                 domain_file,
                                 max_len,
                                 min_len,
                                 output_file,
                                 x_dim=5,
                                 y_dim=5,
                                 rem_individual_files: bool = True,
                                 anonymization: [str, None] = None,
                                 keep_both: bool = False
                                 ):

    problem_dir = os.path.join(data_dir, 'problems_pddl')
    plan_dir = os.path.join(data_dir, 'plans_pddl')
    incorrect_plan_dir = os.path.join(data_dir, 'plans_incorrect')
    json_out_file = os.path.join(data_dir, output_file)

    if anonymization is not None and keep_both:
        not_anonym_data_dir = Path(f'{data_dir}_not_anonym')
        not_anonym_data_dir.mkdir(exist_ok=True)
        not_anonym_domain_file = not_anonym_data_dir / 'domain.pddl'
        not_anonym_dataset_file = not_anonym_data_dir / output_file
        not_anonym_problem_dir = not_anonym_data_dir / 'problems_pddl'
        not_anonym_plan_dir = not_anonym_data_dir / 'plans_pddl'
        not_anonym_inc_plan_dir = not_anonym_data_dir / 'plans_incorrect'
        not_anonym_problem_dir.mkdir(exist_ok=True)
        not_anonym_plan_dir.mkdir(exist_ok=True)
        not_anonym_inc_plan_dir.mkdir(exist_ok=True)

    Path(problem_dir).mkdir(exist_ok=True, parents=True)
    Path(plan_dir).mkdir(exist_ok=True, parents=True)
    Path(incorrect_plan_dir).mkdir(exist_ok=True, parents=True)

    adjacencies, all_locs = create_grid(x=x_dim, y=y_dim)
    goal = create_default_goal(all_locs=all_locs)
    shared_init, adj_dict, state_dict = create_grid_init(adjacencies, all_locs)

    # do everything that is identical for different instances
    objects_str = f'\t(:objects )\n\n'
    goal_str = f'\t(:goal\n\t\t(and\n'
    for fact in goal:
        goal_str += f'\t\t\t{fact}\n'
    goal_str += f'\t\t)\n\t)'

    max_attempts = 4 * n_per_len

    actions, action_prec_and_effects = generate_domain_file_exp(x_dim=x_dim, y_dim=y_dim, write=False)

    if anonymization is None:
        action_mappings = {ac: ac for ac in actions}
    elif anonymization == 'full':
        action_mappings = get_anonymized_action_names(actions=actions)
    else:
        action_mappings = get_mappings_semi_anonymized(actions=actions)

    for plan_len in range(min_len, max_len + 1):
        tasks_so_far = []
        generated = 0
        attempts = 0
        while generated < n_per_len:
            if attempts >= max_attempts:
                break
            print(f'Generating plan length {plan_len}')
            header_str = f'(define (problem light-inst-{next_inst_id})\n\t(:domain lights)\n\n'

            suffix = random.randint(0, 1000000000)
            empty_inst_file_path = os.path.join(TEMP_DIR, f'empty_init_{suffix}.pddl')

            create_empty_initial_state_file(
                all_locs=all_locs,
                goal_str=goal_str,
                header_str=header_str,
                objects_str=objects_str,
                out_file=empty_inst_file_path,
                shared_init=shared_init
            )

            correct_plan_reversed = generate_correct_plan_from_empty(
                domain_file_path=domain_file,
                empty_instance_file_path=empty_inst_file_path,
                plan_len=plan_len
            )
            os.remove(empty_inst_file_path)
            init, correct_plan = generate_init_state_and_plan(
                reversed_plan=correct_plan_reversed,
                all_locs=all_locs,
                action_preconds_and_effects=action_prec_and_effects)

            plan_str = '\n'.join(correct_plan)
            if (init, plan_str) in tasks_so_far:
                continue
            else:
                tasks_so_far.append((init, plan_str))

            print(plan_str)

            init_str = '\t(:init\n'
            for fact in shared_init:
                init_str += f'\t\t{fact}\n'
            for fact in init:
                init_str += f'\t\t{fact}\n'
            init_str += '\t)\n\n'

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

            incorrect_plan, incorrect_action_ind = generate_incomplete_plan(
                incorrect_plan_file=incorrect_plan_path,
                problem_file=instance_path,
                domain_file=domain_file,
                correct_plan_file=plan_path
            )

            correct_data_inst = {
                'problem': problem_str,
                'plan': plan_str,
                'plan_length': plan_len,
                'n_obj': x_dim * y_dim,
                'incorrect_action_index': None
            }
            incorrect_plan_str = '\n'.join(incorrect_plan)
            incorrect_data_inst = {
                'problem': problem_str,
                'plan': incorrect_plan_str,
                'plan_length': len(incorrect_plan),
                'n_obj': x_dim * y_dim,
                'incorrect_action_index': incorrect_action_ind
            }

            if anonymization is not None:
                # Save all the not anonymized data
                if keep_both:

                    with open(not_anonym_dataset_file, 'a') as f:
                        json.dump(correct_data_inst, f)
                        f.write('\n')
                        json.dump(incorrect_data_inst, f)
                        f.write('\n')

                    if not rem_individual_files:
                        shutil.copy(instance_path, not_anonym_problem_dir / f'instance-{next_inst_id}.pddl')
                        shutil.copy(plan_path, not_anonym_plan_dir / f'instance-{next_inst_id}_plan.txt')
                        shutil.copy(incorrect_plan_path, not_anonym_inc_plan_dir / f'instance-{next_inst_id}_incorrect.txt')

                anonymized_plan = anonymize_plan(action_mappings=action_mappings, plan_str=plan_str)
                correct_data_inst = {
                    'problem': problem_str,
                    'plan': anonymized_plan,
                    'plan_length': plan_len,
                    'n_obj': x_dim * y_dim,
                    'incorrect_action_index': None
                }
                anonymized_inc_plan = anonymize_plan(action_mappings=action_mappings, plan_str=incorrect_plan_str)
                incorrect_data_inst = {
                    'problem': problem_str,
                    'plan': anonymized_inc_plan,
                    'plan_length': len(incorrect_plan),
                    'n_obj': x_dim * y_dim,
                    'incorrect_action_index': incorrect_action_ind
                }
                with open(plan_path, 'w') as f:
                    f.write(anonymized_plan)
                with open(incorrect_plan_path, 'w') as f:
                    f.write(anonymized_inc_plan)

            with open(json_out_file, 'a') as f:
                json.dump(correct_data_inst, f)
                f.write('\n')
                json.dump(incorrect_data_inst, f)
                f.write('\n')

            next_inst_id += 1
            generated += 1
            attempts += 1

            if rem_individual_files:
                os.remove(instance_path)
                os.remove(plan_path)
                os.remove(incorrect_plan_path)

    if anonymization is not None:

        with open(domain_file, 'r') as f:
            domain_str = f.read()

        for orig_ac, new_ac in action_mappings.items():
            assert orig_ac in domain_str
            domain_str = domain_str.replace(orig_ac, new_ac)

        new_domain_file = domain_file.replace('.pddl', f'_{anonymization}.pddl')
        with open(new_domain_file, 'w') as f:
            f.write(domain_str)
