import os
import random
import itertools
import pymimir as mm
from copy import copy
from planning_utils.execute_plan import execute_plan_mimir
from planning_utils.validate_plan import run_validation
from planning_utils.pddl_processing import process_problem_file


def get_correct_plan_well_formed(domain_file_path,
                                 problem_file_path,
                                 ball_locs: dict,
                                 robot_state: dict,
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

    prev_action = None

    prev_appl_actions = action_mappings
    prev_state = current_state
    actions_to_remove = []
    while len(correct_plan) < target_plan_len:
        current_iter += 1
        if current_iter >= max_iterations:
            raise ValueError

        next_action = get_random_appl_action(all_appl_actions=all_appl_actions_str, actions_to_remove=actions_to_remove)

        current_state = state_repository.get_or_create_successor_state(current_state, action_mappings[next_action])
        all_applicable_actions_mimir = successor_generator.compute_applicable_actions(current_state)
        all_appl_actions_str = [action.to_string_for_plan(factories) for action in all_applicable_actions_mimir]
        action_mappings = {ac_str: ac_mm for (ac_str, ac_mm) in zip(all_appl_actions_str, all_applicable_actions_mimir)}

        # Go one step back if dead-end
        if len(all_applicable_actions_mimir) == 0:
            current_state = prev_state
            action_mappings = prev_appl_actions
            all_appl_actions_str = list(action_mappings.keys())
            actions_to_remove.append(next_action)
            continue

        correct_plan.append(next_action)
        actions_to_remove = []

        prev_action = next_action
        prev_state = current_state
        prev_appl_actions = action_mappings

        args = prev_action.replace('(', '').replace(')', '').split(' ')[1:]
        if 'pick' in next_action:
            ball = args[0]
            gripper = args[2]
            ball_locs[ball] = gripper
            robot_state[gripper] = 'carry'
        elif 'drop' in next_action:
            ball = args[0]
            room = args[1]
            gripper = args[2]
            ball_locs[ball] = room
            robot_state[gripper] = 'free'
        elif 'move' in next_action:
            room2 = args[1]
            robot_state['room'] = room2
        else:
            raise ValueError

    last_action_type = correct_plan[-1].replace('(', '').replace(')', '').split(' ')[0]
    last_action_pick = True if 'pick' in last_action_type else False
    carrying = []
    if robot_state['gripper0'] == 'carry':
        carrying.append('gripper0')
    if robot_state['gripper1'] == 'carry':
        carrying.append('gripper1')

    if not len(carrying):
        for ball, ball_lo in ball_locs.items():
            assert ball_lo.startswith('room')
        return correct_plan, ball_locs

    if last_action_pick and len(carrying) == 1:
        # Remove the last pick-up action
        args = prev_action.replace('(', '').replace(')', '').split(' ')[1:]
        ball = args[0]
        room = args[1]
        gripper = args[2]
        assert ball_locs[ball] == gripper
        correct_plan = correct_plan[:-1]  # remove last pick-up
        ball_locs[ball] = room
        return correct_plan, ball_locs

    elif len(carrying) == 1:
        next_action = random.choice(all_appl_actions_str)
        if not 'drop' in next_action:
            return None, None

        else:
            args = next_action.replace('(', '').replace(')', '').split(' ')[1:]
            ball = args[0]
            room = args[1]
            ball_locs[ball] = room
            correct_plan.append(next_action)
            return correct_plan, ball_locs
    else:
        return None, None


def get_random_appl_action(all_appl_actions, actions_to_remove):
    pick_actions = []
    move_actions = []
    drop_actions = []
    possible_action_types = set()

    for action in all_appl_actions:
        if action in actions_to_remove:
            continue
        if 'move' in action:
            move_actions.append(action)
            possible_action_types.add('move')
        elif 'pick' in action:
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

    return next_action


def get_incomplete_plan(domain_file,
                        problem_file_path,
                        plan_file_path,
                        incorrect_file_path):

    with open(plan_file_path, 'r') as f:
        plan = []
        for line in f.readlines():
            if line.strip().startswith(';'):
                continue
            elif line.strip() == '(PLAN_END)':
                continue
            plan.append(line.strip())

    if len(plan) < 4:
        print(f'Not possible for {plan_file_path}')
        return None, None

    last_drop_action_id = None
    orig_drop_action = None
    for step, action in enumerate(plan):
        if 'drop' in action:
            last_drop_action_id = step
            orig_drop_action = action

    assert last_drop_action_id is not None

    incomplete_plan = plan[:last_drop_action_id]

    max_additions = 4
    already_added = 0
    dropped = False
    actions_not_to_take = [orig_drop_action]
    while not dropped:
        if already_added >= max_additions:
            raise AssertionError
        current_state, all_appl_actions_str = execute_plan_mimir(
            domain_file=domain_file,
            problem_file=problem_file_path,
            plan=incomplete_plan
        )
        # avoid picking up a ball again or just sampling the same drop action
        candidate_actions = [ac for ac in all_appl_actions_str if 'pick' not in ac]
        candidate_actions = [ac for ac in candidate_actions if ac not in actions_not_to_take]

        if len(candidate_actions):
            # try to drop if possible
            drop_cand_actions = [ac for ac in candidate_actions if 'drop' in ac]
            if len(drop_cand_actions):
                next_action = random.choice(drop_cand_actions)
            else:
                next_action = random.choice(candidate_actions)
            incomplete_plan.append(next_action)
            if 'drop' in next_action:
                dropped = True
            else:
                already_added += 1
        else:
            incomplete_plan = incomplete_plan[:-1]
            assert len(incomplete_plan)
            actions_not_to_take.append(incomplete_plan[-1])
        if dropped:
            break

    with open(incorrect_file_path, 'w') as f:
        f.write('\n'.join(incomplete_plan))

    reached_goal, executable = run_validation(domain_file=domain_file,
                                              problem_file=problem_file_path,
                                              plan_file=incorrect_file_path,
                                              plan_end_tag=False)

    assert executable
    assert not reached_goal

    return incomplete_plan, -1


def get_incorrect_plan(domain_file,
                       plan_file_path,
                       incorrect_file_path,
                       problem_file_path):

    objects, _, _, _, _ = process_problem_file(domain_path=domain_file,
                                               problem_path=problem_file_path)
    balls = [obj for obj in objects if obj.startswith('ball')]
    rooms = [obj for obj in objects if obj.startswith('room')]

    with open(plan_file_path, 'r') as f:
        plan = []
        for line in f.readlines():
            if line.strip().startswith(';'):
                continue
            elif line.strip() == '(PLAN_END)':
                continue
            plan.append(line.strip())

    attempts = 10
    found_incorrect = False

    for _ in range(attempts):

        incorrect_ind = len(plan) - 1

        plan_incorrect = copy(plan)

        incorrect_action = get_incorrect_action(
            domain_file_path=domain_file,
            instance_file_path=problem_file_path,
            plan=plan,
            action_ind=incorrect_ind,
            balls=balls,
            rooms=rooms)
        if incorrect_action is None:
            continue

        plan_incorrect[incorrect_ind] = incorrect_action

        with open(incorrect_file_path, 'w') as f:
            f.write('\n'.join(plan_incorrect))

        reached_goal, executable = run_validation(domain_file=domain_file,
                                                  problem_file=problem_file_path,
                                                  plan_file=incorrect_file_path,
                                                  plan_end_tag=False)

        if not reached_goal or not executable:
            assert not executable
            found_incorrect = True
            break

    if not found_incorrect:
        try:
            os.remove(incorrect_file_path)
        except FileNotFoundError:
            pass
        raise AssertionError

    else:
        return plan_incorrect, incorrect_ind


def get_incorrect_action(domain_file_path,
                         instance_file_path,
                         plan,
                         action_ind,
                         rooms,
                         balls):
    parser = mm.PDDLParser(str(domain_file_path), str(instance_file_path))
    problem = parser.get_problem()
    factories = parser.get_pddl_factories()

    successor_generator = mm.LiftedApplicableActionGenerator(problem, factories)
    state_repository = mm.StateRepository(successor_generator)
    current_state = state_repository.get_or_create_initial_state()
    all_applicable_actions_mimir = successor_generator.compute_applicable_actions(current_state)
    all_appl_actions_str = [action.to_string_for_plan(factories) for action in all_applicable_actions_mimir]
    action_mappings = {ac_str: ac_mm for (ac_str, ac_mm) in zip(all_appl_actions_str, all_applicable_actions_mimir)}

    plan_to_execute = plan[:action_ind]
    original_action = plan[action_ind]
    for action in plan_to_execute:
        current_state = state_repository.get_or_create_successor_state(current_state, action_mappings[action])
        all_applicable_actions_mimir = successor_generator.compute_applicable_actions(current_state)
        all_appl_actions_str = [action.to_string_for_plan(factories) for action in all_applicable_actions_mimir]
        action_mappings = {ac_str: ac_mm for (ac_str, ac_mm) in zip(all_appl_actions_str, all_applicable_actions_mimir)}

    all_applicable_actions_mimir = successor_generator.compute_applicable_actions(current_state)
    all_appl_actions_str = [action.to_string_for_plan(factories) for action in all_applicable_actions_mimir]

    # if last action make sure it's a valid last action of a plan
    incorrect_action = _get_random_not_pick_action(
        rooms=rooms,
        balls=balls,
        all_appl_actions_str=all_appl_actions_str
    )

    return incorrect_action


def _get_random_not_pick_action(rooms, balls, all_appl_actions_str: list):
    all_room_combinations = list(itertools.product(rooms, rooms))
    all_drop_pick_combinations = list(itertools.product(rooms, balls, ['gripper0', 'gripper1']))

    actions = set()
    for rand_rooms in all_room_combinations:
        action = f'(move {rand_rooms[0]} {rand_rooms[1]})'
        actions.add(action)
    for (rand_room, rand_ball, rand_grip) in all_drop_pick_combinations:
        action = f'(drop {rand_ball} {rand_room} {rand_grip})'
        actions.add(action)

    while len(actions):
        candidate_action = random.choice(list(actions))
        if candidate_action not in all_appl_actions_str:
            return candidate_action
        actions.remove(candidate_action)

    return None
