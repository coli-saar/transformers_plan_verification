import os
import json
import math
from copy import deepcopy
from typing import Union
from pathlib import Path
from collections import defaultdict
from data_generation.grippers_heavy.grippers_heavy_helpers import *
from data_generation.grippers_heavy.generate_plans_df import generate_correct_plan_df, get_incomplete_plan_df, get_incorrect_plan_df
from data_generation.grippers_heavy.generate_plans_wf import get_correct_plan_well_formed, get_incomplete_plan, get_incorrect_plan
from planning_utils.validate_plan import run_validation


def generate_instance(n_balls,
                      n_rooms,
                      min_ratio_heavy,
                      max_ratio_heavy,
                      target_plan_len,
                      domain_file_path,
                      problem_file_path,
                      plan_file_path,
                      delete_relaxed: bool,
                      well_formed: bool,
                      max_rooms_id: int,
                      max_balls_id: int,
                      delete_free_param: Union[dict, None]
                      ):
    """

    :param n_balls:
    :param n_rooms:
    :param target_plan_len:
    :param domain_file_path:
    :param problem_file_path:
    :param plan_file_path:
    :param delete_relaxed:
    :param well_formed:
    :param max_rooms_id:
    :param max_balls_id:
    :param delete_free_param: if delete free and correct plan should be based on well-formed, set this
    :return:
    """

    grippers, rooms, balls = init_objects(n_balls=n_balls,
                                          n_rooms=n_rooms,
                                          max_rooms_id=max_rooms_id,
                                          max_balls_id=max_balls_id)
    min_heavy = math.floor(n_balls * min_ratio_heavy)
    max_heavy = math.floor(n_balls * max_ratio_heavy)
    n_heavy_balls = random.randint(min_heavy, max_heavy)
    heavy_balls = random.sample(balls, k=n_heavy_balls)

    init_state, balls_init, robot_init = init_state_default(
        all_rooms=rooms,
        all_balls=balls,
        heavy_balls=heavy_balls)

    dummy_goal, _ = goal_state_default(all_rooms=rooms, all_balls=balls)
    temp_problem_file = problem_file_path
    write_problem(objects=[grippers, rooms, balls],
                  init_state=init_state,
                  goal_state=dummy_goal,
                  save_path=temp_problem_file)

    action_to_ignore = None
    action_to_replace = None
    add_fixed_goal_ind = None

    if delete_relaxed:
        version = 'df'
        domain_file_wf = delete_free_param['domain_file_wf']
        n_contrasting_actions = delete_free_param.get(f'n_{version}_only_actions', None)
        if n_contrasting_actions is None:
            n_contrasting_actions = get_n_actions_contrasting(
                contrasting_param=delete_free_param,
                version=version,
                target_plan_len=target_plan_len
            )

        correct_plan, balls_goal, action_to_ignore, action_to_replace, add_fixed_goal_ind = generate_correct_plan_df(
            domain_file_path=domain_file_path,
            domain_file_path_wf=domain_file_wf,
            problem_file_path=temp_problem_file,
            orig_ball_locs=deepcopy(balls_init),
            orig_robot_state=deepcopy(robot_init),
            target_plan_len=target_plan_len,
            n_df_only_actions=n_contrasting_actions,
            rooms=rooms
        )

    elif well_formed:
        correct_plan, balls_goal = get_correct_plan_well_formed(
            domain_file_path=domain_file_path,
            problem_file_path=problem_file_path,
            ball_locs=deepcopy(balls_init),
            robot_state=deepcopy(robot_init),
            target_plan_len=target_plan_len
        )
    else:
        raise ValueError

    if correct_plan is None:
        return correct_plan, balls_goal, action_to_ignore, action_to_replace, add_fixed_goal_ind

    all_balls_init = list(balls_init.keys())
    all_balls_goal = list(balls_goal.keys())
    all_balls_init.sort()
    all_balls_goal.sort()
    assert all_balls_goal == all_balls_init

    goal_state = []
    for ball, goal_loc in balls_goal.items():
        goal_state.append(f'(at {ball} {goal_loc})')

    problem_str = write_problem(
        objects=[grippers, rooms, balls],
        init_state=init_state,
        goal_state=goal_state,
        save_path=problem_file_path)

    with open(plan_file_path, 'w') as f:
        f.write('\n'.join(correct_plan))

    reached_goal, executable = run_validation(domain_file=domain_file_path,
                                              problem_file=problem_file_path,
                                              plan_file=plan_file_path,
                                              plan_end_tag=False)
    if not reached_goal or not executable:
        return None, None, None, None, None
    assert reached_goal and executable

    return problem_str, correct_plan, action_to_ignore, action_to_replace, add_fixed_goal_ind


def generate_dataset(min_ratio_ball, max_ratio_ball,
                     min_ratio_room, max_ratio_room,
                     min_ratio_heavy, max_ratio_heavy,
                     n_per_len,
                     min_len,
                     max_len,
                     dataset_dir,
                     well_formed: bool,
                     delete_relaxed: bool,
                     output_file,
                     domain_file,
                     incomplete: bool,
                     max_balls_id,
                     max_rooms_id,
                     next_inst_id=0,
                     rem_individual_files=True,
                     delete_free_param: Union[dict, None] = None):


    problem_dir = os.path.join(dataset_dir, 'problems_pddl')
    plan_dir = os.path.join(dataset_dir, 'plans_pddl')
    incorrect_plan_dir = os.path.join(dataset_dir, 'plans_incorrect')
    Path(problem_dir).mkdir(exist_ok=True, parents=True)
    Path(plan_dir).mkdir(exist_ok=True)
    Path(incorrect_plan_dir).mkdir(exist_ok=True)

    json_out_file = os.path.join(dataset_dir, output_file)
    other_lens_dir = os.path.join(dataset_dir, 'other_lens')
    Path(other_lens_dir).mkdir(exist_ok=True)
    json_out_other_lens = os.path.join(other_lens_dir, output_file.replace('.jsonl', '_other_lens.jsonl'))

    dataset = []
    required_plan_lens = []
    for target_plan_len in range(min_len, max_len + 1):
        required_plan_lens.extend([target_plan_len] * n_per_len)

    tasks_so_far = defaultdict(list)
    while len(required_plan_lens) != 0:

        orig_target_plan_len = required_plan_lens.pop(0)
        candidate_lens_smaller = [i for i in range(orig_target_plan_len - 3, orig_target_plan_len)]
        candidate_lens_larger = [i for i in range(orig_target_plan_len + 1, orig_target_plan_len + 4)]
        candidate_plan_lens = [orig_target_plan_len]

        # add some even further plan lens
        more_distant_lens_smaller = [i for i in range(orig_target_plan_len - 5, orig_target_plan_len - 3)]
        more_distant_lens_larger = [i for i in range(orig_target_plan_len + 4, orig_target_plan_len + 6)]

        lens_to_try = []
        for _ in range(20):
            lens_to_try.extend(candidate_plan_lens * 2)
            lens_to_try.extend(candidate_lens_larger)
            lens_to_try.extend(candidate_lens_smaller)
            lens_to_try.extend(more_distant_lens_larger)
            lens_to_try.extend(more_distant_lens_smaller)

        while len(lens_to_try):
            target_plan_len = lens_to_try.pop(0)

            n_balls, n_rooms = get_number_objects_ratios(
                target_plan_len=target_plan_len,
                min_ratio_room=min_ratio_room, max_ratio_room=max_ratio_room,
                min_ratio_ball=min_ratio_ball, max_ratio_ball=max_ratio_ball,
                max_rooms_id=max_rooms_id, max_balls_id=max_balls_id
            )

            print(f'Generating plans of length {orig_target_plan_len}')

            problem_file = os.path.join(problem_dir, f'instance-{next_inst_id}.pddl')
            plan_file = os.path.join(plan_dir, f'instance-{next_inst_id}_plan.txt')
            incorrect_plan_file = os.path.join(incorrect_plan_dir, f'instance-{next_inst_id}_incorrect.txt')

            problem_str, correct_plan, action_to_ignore, action_to_replace, add_fixed_goal_ind = generate_instance(
                domain_file_path=domain_file,
                problem_file_path=problem_file,
                plan_file_path=plan_file,
                n_balls=n_balls,
                n_rooms=n_rooms,
                max_ratio_heavy=max_ratio_heavy,
                min_ratio_heavy=min_ratio_heavy,
                target_plan_len=target_plan_len,
                max_rooms_id=max_rooms_id,
                max_balls_id=max_balls_id,
                delete_relaxed=delete_relaxed,
                well_formed=well_formed,
                delete_free_param=delete_free_param
            )
            if problem_str is None:
                continue

            actual_plan_len = len(correct_plan)
            # If already enough there: continue with next
            if len(tasks_so_far[actual_plan_len]) == n_per_len:
                os.remove(problem_file)
                os.remove(plan_file)
                continue

            plan_str = '\n'.join(correct_plan)
            if (problem_str, plan_str) in tasks_so_far[actual_plan_len]:
                continue
            else:
                tasks_so_far[actual_plan_len].append((problem_str, plan_str))

            # generate incorrect plan and validate incorrect plan
            if incomplete:
                try:
                    if delete_relaxed:
                        incorrect_plan, incorrect_ind = get_incomplete_plan_df(
                            domain_file=domain_file,
                            problem_file_path=problem_file,
                            plan_file_path=plan_file,
                            incorrect_file_path=incorrect_plan_file,
                            add_fixed_goal_ind=add_fixed_goal_ind
                        )
                    else:
                        incorrect_plan, incorrect_ind = get_incomplete_plan(
                            domain_file=domain_file,
                            problem_file_path=problem_file,
                            plan_file_path=plan_file,
                            incorrect_file_path=incorrect_plan_file
                        )
                except AssertionError:
                    tasks_so_far[actual_plan_len].remove((problem_str, plan_str))
                    continue
            else:
                try:
                    if delete_relaxed:
                        incorrect_plan, incorrect_ind = get_incorrect_plan_df(
                            domain_file=domain_file,
                            problem_file_path=problem_file,
                            plan_file_path=plan_file,
                            incorrect_file_path=incorrect_plan_file,
                            action_to_replace=action_to_replace
                        )
                    else:
                        incorrect_plan, incorrect_ind = get_incorrect_plan(
                            domain_file=domain_file,
                            problem_file_path=problem_file,
                            plan_file_path=plan_file,
                            incorrect_file_path=incorrect_plan_file
                        )
                except AssertionError:
                    tasks_so_far[actual_plan_len].remove((problem_str, plan_str))
                    continue

            correct_data_inst = {
                'problem': problem_str,
                'plan': '\n'.join(correct_plan),
                'plan_length': len(correct_plan),
                'n_obj': n_balls + n_rooms + 2,
                'incorrect_action_index': None
            }

            incorrect_data_inst = {
                'problem': problem_str,
                'plan': '\n'.join(incorrect_plan),
                'plan_length': len(incorrect_plan),
                'n_obj': n_balls + n_rooms + 2,
                'incorrect_action_index': incorrect_ind
            }

            if actual_plan_len < min_len or actual_plan_len > max_len:
                with open(json_out_other_lens, 'a') as f:
                    json.dump(correct_data_inst, f)
                    f.write('\n')
                    json.dump(incorrect_data_inst, f)
                    f.write('\n')
                os.remove(problem_file)
                os.remove(plan_file)
                os.remove(incorrect_plan_file)
                continue

            if actual_plan_len != orig_target_plan_len:
                if actual_plan_len in required_plan_lens:
                    required_plan_lens.remove(actual_plan_len)

            dataset.append(correct_data_inst)
            dataset.append(incorrect_data_inst)

            next_inst_id += 1

            with open(json_out_file, 'a') as f:
                json.dump(correct_data_inst, f)
                f.write('\n')
                json.dump(incorrect_data_inst, f)
                f.write('\n')

            if rem_individual_files:
                os.remove(problem_file)
                os.remove(plan_file)
                os.remove(incorrect_plan_file)

            if orig_target_plan_len == actual_plan_len:
                break

    missing_vals = dict()
    for pl in range(min_len, max_len + 1):
        n_tasks = len(tasks_so_far[pl])
        if n_tasks != n_per_len:
            missing_vals[pl] = n_per_len - n_tasks
    print(missing_vals)
