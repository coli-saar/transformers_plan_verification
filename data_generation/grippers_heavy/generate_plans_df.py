import os
import random
from copy import copy, deepcopy
import pymimir as mm
from planning_utils.execute_plan import execute_plan_mimir
from planning_utils.pddl_processing import process_problem_file
from planning_utils.validate_plan import run_validation


def generate_correct_plan_df(domain_file_path,
                             domain_file_path_wf,
                             problem_file_path,
                             orig_ball_locs: dict,
                             rooms: list,
                             orig_robot_state: dict,
                             target_plan_len: int,
                             n_df_only_actions: int):
    correct_plan_wf, actions_to_remove, action_to_ignore = None, None, None

    for _ in range(10):
        if n_df_only_actions == target_plan_len - 2:
            break
        correct_plan_wf, ball_locs, action_to_ignore, action_to_replace, actions_to_remove, add_fixed_goal_ind = get_correct_plan_random(
            domain_file_path=domain_file_path_wf,
            problem_file_path=problem_file_path,
            ball_locs=deepcopy(orig_ball_locs),
            robot_state=deepcopy(orig_robot_state),
            target_plan_len=target_plan_len - n_df_only_actions,
            rooms=rooms
        )
        if correct_plan_wf is None:
            n_df_only_actions += 1
        else:
            break

    if correct_plan_wf is None:
        return None, None, None, None, None

    if len(correct_plan_wf) != target_plan_len - n_df_only_actions:
        target_plan_len = len(correct_plan_wf) + n_df_only_actions

    action_type_indices = [True] * n_df_only_actions + [False] * (target_plan_len - n_df_only_actions - 1)
    random.shuffle(action_type_indices)
    action_type_indices.append(False)

    remaining_actions = copy(correct_plan_wf)
    correct_plan_df = []
    for ind in range(target_plan_len):
        if action_type_indices[ind]:
            correct_plan_df.append(None)
        else:
            next_ac = remaining_actions.pop(0)
            correct_plan_df.append(next_ac)

    final_correct_plan = []
    for step, action in enumerate(correct_plan_df):
        if action is not None:
            final_correct_plan.append(action)
            continue

        # execute up until current action
        plan_to_execute = correct_plan_df[:step]
        plan_to_execute = [ac for ac in plan_to_execute if ac is not None]

        _, all_appl_actions_df = execute_plan_mimir(
            domain_file=domain_file_path,
            problem_file=problem_file_path,
            plan=plan_to_execute
        )

        try:
            _, all_appl_actions_wf = execute_plan_mimir(
                domain_file=domain_file_path_wf,
                problem_file=problem_file_path,
                plan=plan_to_execute
            )
        except KeyError:
            print(correct_plan_wf)
            print(plan_to_execute)
            _, all_appl_actions_wf = execute_plan_mimir(
                domain_file=domain_file_path_wf,
                problem_file=problem_file_path,
                plan=plan_to_execute
            )
        assert len(all_appl_actions_df)
        assert len(all_appl_actions_wf)

        candidate_actions = [ac for ac in all_appl_actions_df if ac not in actions_to_remove]
        for action in candidate_actions:
            ac_name, ac_args = parse_action(action)
            if 'move' in action:
                r1, r2 = ac_args
                if action_to_ignore[0] == 'move' and action_to_ignore[1] == r2:
                    if action in candidate_actions:
                        candidate_actions.remove(action)
            elif 'pick' in action:
                ball, room, gripper = ac_args
                if action_to_ignore[0] == 'pick' and action_to_ignore[1] == ball and action_to_ignore[2] == gripper:
                    if action in candidate_actions:
                        candidate_actions.remove(action)

        candidate_actions_df_only = [ac for ac in candidate_actions if ac not in all_appl_actions_wf]
        if len(candidate_actions_df_only) == 0:
            return None, None, None, None, None

        next_action, _ = distribution_probs_differently_types(
            all_appl_actions=candidate_actions_df_only,
            actions_to_remove=[],
            action_to_ignore=['', '']
        )
        final_correct_plan.append(next_action)

    return final_correct_plan, ball_locs, action_to_ignore, action_to_replace, add_fixed_goal_ind


def get_correct_plan_random(domain_file_path,
                            problem_file_path,
                            ball_locs: dict,
                            robot_state: dict,
                            rooms: list,
                            target_plan_len):
    correct_plan = []
    parser = mm.PDDLParser(str(domain_file_path), str(problem_file_path))
    problem = parser.get_problem()
    factories = parser.get_pddl_factories()

    successor_generator = mm.LiftedApplicableActionGenerator(problem, factories)
    state_repository = mm.StateRepository(successor_generator)
    current_state = state_repository.get_or_create_initial_state()
    all_applicable_actions_mimir = successor_generator.compute_applicable_actions(current_state)
    all_appl_actions_str = [action.to_string_for_plan(factories) for action in all_applicable_actions_mimir]
    action_mappings = {ac_str: ac_mm for (ac_str, ac_mm) in zip(all_appl_actions_str, all_applicable_actions_mimir)}

    max_iterations = target_plan_len * 10
    current_iter = 0

    relevant_action_type = random.choice(['pick', 'move'])

    if relevant_action_type == 'pick':
        grippers = ['gripper0', 'gripper1']
        random.shuffle(grippers)
        g1, g2 = grippers
        ball = random.choice(list(ball_locs.keys()))
        action_to_ignore = ['pick', ball, g1]
        action_to_replace = ['drop', ball, g1]
    else:
        target_room = random.choice(rooms)
        action_to_ignore = ['move', target_room]
        action_to_replace = ['move', target_room]

    fixed_goal_ball = random.choice(list(ball_locs.keys()))
    add_fixed_goal_ind = None

    actions_to_remove = []
    while len(correct_plan) < target_plan_len:
        current_iter += 1
        if current_iter >= max_iterations:
            raise ValueError

        try:
            next_action, actions_to_remove = distribution_probs_differently_types(
                all_appl_actions=all_appl_actions_str,
                actions_to_remove=actions_to_remove,
                action_to_ignore=action_to_ignore
            )
        except IndexError:
            return None, None, None, None, None, None

        current_state = state_repository.get_or_create_successor_state(current_state, action_mappings[next_action])
        all_applicable_actions_mimir = successor_generator.compute_applicable_actions(current_state)
        all_appl_actions_str = [action.to_string_for_plan(factories) for action in all_applicable_actions_mimir]
        action_mappings = {ac_str: ac_mm for (ac_str, ac_mm) in zip(all_appl_actions_str, all_applicable_actions_mimir)}

        # Go one step back if dead-end
        if len(all_applicable_actions_mimir) == 0:
            return None, None, None, None, None, None

        correct_plan.append(next_action)

        prev_action = next_action

        args = prev_action.replace('(', '').replace(')', '').split(' ')[1:]
        if 'pick' in next_action:
            gripper = args[2]
            robot_state[gripper] = 'carry'
        elif 'drop' in next_action:
            ball = args[0]
            room = args[1]
            gripper = args[2]
            ball_locs[ball] = room
            robot_state[gripper] = 'free'

            if ball == fixed_goal_ball:
                add_fixed_goal_ind = len(correct_plan) - 1
                # remove actions moving that ball
                for room in rooms:
                    actions_to_remove.extend([
                        f'(drop {fixed_goal_ball} {room} gripper0)',
                        f'(drop {fixed_goal_ball} {room} gripper1)'
                    ])

        elif 'move' in next_action:
            room2 = args[1]
            robot_state['room'] = room2
        else:
            raise ValueError

    last_action_type = correct_plan[-1].replace('(', '').replace(')', '').split(' ')[0]
    last_action_pick = True if 'pick' in last_action_type else False

    if add_fixed_goal_ind is None:
        return None, None, None, None, None, None

    if last_action_pick:
        # Remove the last pick-up action
        correct_plan = correct_plan[:-1]  # remove last pick-up

    last_action_type = correct_plan[-1].replace('(', '').replace(')', '').split(' ')[0]
    last_action_pick = True if 'pick' in last_action_type else False
    if last_action_pick:
        found = False
        for attempt in range(6):
            next_action = random.choice(all_appl_actions_str)

            if 'pick' in next_action:
                continue
            elif 'move' in next_action:
                correct_plan.append(next_action)
                found = True
                break
            else:
                args = next_action.replace('(', '').replace(')', '').split(' ')[1:]
                ball = args[0]
                room = args[1]
                ball_locs[ball] = room
                correct_plan.append(next_action)
                found = True
                break

        if not found:
            return None, None, None, None, None, None

    return correct_plan, ball_locs, action_to_ignore, action_to_replace, actions_to_remove, add_fixed_goal_ind


def parse_action(expr):
    """Parse '(predicate arg1 arg2)' into ('predicate', [args])"""
    tokens = expr.strip("()").split()
    return tokens[0], tokens[1:]


def distribution_probs_differently_types(all_appl_actions,
                                         actions_to_remove,
                                         action_to_ignore):
    pick_actions = []
    move_actions = []
    drop_actions = []
    possible_action_types = set()

    for action in all_appl_actions:
        ac_name, ac_args = parse_action(action)

        if action in actions_to_remove:
            continue

        if 'move' in action:
            r1, r2 = ac_args
            if action_to_ignore[0] == 'move' and action_to_ignore[1] == r2:
                actions_to_remove.append(action)
            else:
                move_actions.append(action)
                possible_action_types.add('move')
        elif 'pick' in action:
            ball, room, gripper = ac_args
            if action_to_ignore[0] == 'pick' and action_to_ignore[1] == ball and action_to_ignore[2] == gripper:
                actions_to_remove.append(action)
            else:
                pick_actions.append(action)
                possible_action_types.add('pick')
        elif 'drop' in action:
            drop_actions.append(action)
            possible_action_types.add('drop')
        else:
            raise ValueError

    action_type = random.choice(list(possible_action_types))
    if action_type == 'move':
        candidate_actions = move_actions
    elif action_type == 'pick':
        candidate_actions = pick_actions
    elif action_type == 'drop':
        candidate_actions = drop_actions
    else:
        raise ValueError

    assert len(candidate_actions)
    next_action = random.choice(candidate_actions)
    actions_to_remove = list(set(actions_to_remove))

    return next_action, actions_to_remove


def get_incorrect_plan_df(domain_file,
                          problem_file_path,
                          plan_file_path,
                          incorrect_file_path,
                          action_to_replace):
    objects, _, _, _, _ = process_problem_file(domain_path=domain_file,
                                               problem_path=problem_file_path)
    balls = [obj for obj in objects if obj.startswith('ball')]
    rooms = [obj for obj in objects if obj.startswith('room')]
    grippers = [obj for obj in objects if obj.startswith('gripper')]

    with open(plan_file_path, 'r') as f:
        plan = []
        for line in f.readlines():
            if line.strip().startswith(';'):
                continue
            elif line.strip() == '(PLAN_END)':
                continue
            plan.append(line.strip())

    if action_to_replace[0] == 'drop':
        target_room = random.choice(rooms)
        assert action_to_replace[1] in balls
        assert action_to_replace[2] in grippers
        action_to_replace = f'(drop {action_to_replace[1]} {target_room} {action_to_replace[2]})'
    else:
        room = action_to_replace[1]
        assert room in rooms
        version = random.choice(['drop', 'move'])
        if version == 'drop':
            gripper = random.choice(grippers)
            ball = random.choice(balls)
            action_to_replace = f'(drop {ball} {room} {gripper})'
        else:
            to_room = random.choice(rooms)
            action_to_replace = f'(move {room} {to_room})'

    plan_incorrect = plan[:-1] + [action_to_replace]

    assert plan_incorrect != plan

    with open(incorrect_file_path, 'w') as f:
        f.write('\n'.join(plan_incorrect))

    reached_goal, executable = run_validation(domain_file=domain_file,
                                              problem_file=problem_file_path,
                                              plan_file=incorrect_file_path,
                                              plan_end_tag=False)
    found_incorrect = False
    if not reached_goal or not executable:
        assert not executable
        found_incorrect = True

    if not found_incorrect:
        try:
            os.remove(incorrect_file_path)
        except FileNotFoundError:
            pass
        raise AssertionError

    else:
        return plan_incorrect, -1


def get_incomplete_plan_df(domain_file,
                           problem_file_path,
                           plan_file_path,
                           incorrect_file_path,
                           add_fixed_goal_ind):
    with open(plan_file_path, 'r') as f:
        plan = []
        for line in f.readlines():
            if line.strip().startswith(';'):
                continue
            elif line.strip() == '(PLAN_END)':
                continue
            plan.append(line.strip())

    incomplete_plan = copy(plan)
    incomplete_plan.pop(add_fixed_goal_ind)
    with open(incorrect_file_path, 'w') as f:
        f.write('\n'.join(incomplete_plan))

    reached_goal, executable = run_validation(domain_file=domain_file,
                                              problem_file=problem_file_path,
                                              plan_file=incorrect_file_path,
                                              plan_end_tag=False)
    assert executable
    assert not reached_goal

    return incomplete_plan, -1
