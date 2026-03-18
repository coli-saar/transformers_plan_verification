import random


def init_state_default(all_rooms, all_balls, heavy_balls):
    init_state = []

    room_for_robot = random.choice(all_rooms)
    init_state.append(f'(at-robby {room_for_robot})')
    init_state.append('(gripper gripper0)')
    init_state.append('(gripper gripper1)')
    init_state.append(f'(free gripper0)')
    init_state.append(f'(free gripper1)')
    robot_init = {'room': room_for_robot,
                  'gripper0': 'free',
                  'gripper1': 'free'}

    for room in all_rooms:
        init_state.append(f'(room {room})')

    balls_init = dict()
    for ball in all_balls:
        init_state.append(f'(ball {ball})')
    for ball in all_balls:
        room_for_ball = random.choice(all_rooms)
        balls_init[ball] = room_for_ball
        init_state.append(f'(at {ball} {room_for_ball})')

    for ball in heavy_balls:
        init_state.append(f'(heavy {ball})')

    return init_state, balls_init, robot_init


def goal_state_default(all_rooms,
                       all_balls):
    balls_goal = dict()
    goal_state = []
    for ball in all_balls:
        room_for_ball = random.choice(all_rooms)
        balls_goal[ball] = room_for_ball
        goal_state.append(f'(at {ball} {room_for_ball})')

    return goal_state, balls_goal


def init_objects(n_rooms: int,
                 n_balls: int,
                 max_rooms_id: int = 400,
                 max_balls_id: int = 400):

    all_grippers = ['gripper0', 'gripper1']
    all_rooms = []
    all_balls = []

    potential_ball_ids = [i for i in range(0, max_balls_id)]
    potential_room_ids = [i for i in range(0, max_rooms_id)]

    ball_ids = random.sample(potential_ball_ids, k=n_balls)
    room_ids = random.sample(potential_room_ids, k=n_rooms)
    ball_ids.sort()
    room_ids.sort()

    for room_id in room_ids:
        all_rooms.append(f'room_{room_id}')

    for ball_id in ball_ids:
        all_balls.append(f'ball_{ball_id}')

    return all_grippers, all_rooms, all_balls


def write_problem(
        objects: list,
        init_state: list,
        goal_state: list,
        save_path
    ):
    all_grippers = objects[0]
    all_rooms = objects[1]
    all_balls = objects[2]

    content = f'(define (problem grippers-{len(all_rooms)}-{len(all_balls)})\n'
    content += f'(:domain gripper-strips)\n'
    content += f'(:objects\n'

    gripp_str = ' '.join(all_grippers)
    room_str = ' '.join(all_rooms)
    ball_str = ' '.join(all_balls)

    content += f'\t{gripp_str}\n\t{room_str}\n\t{ball_str}\n)\n'

    content += f'(:init\n'
    for pred in init_state:
        content += f'\t{pred}\n'
    content += ')\n'

    content += f'(:goal\n\t(and\n'
    for pred in goal_state:
        content += f'\t\t{pred}\n'
    content += ')\n)\n'

    content += ')\n'

    with open(save_path, 'w') as f:
        f.write(content)

    return content


def get_number_objects_ratios(target_plan_len,
                              min_ratio_ball, max_ratio_ball,
                              min_ratio_room, max_ratio_room,
                              max_rooms_id,
                              max_balls_id):

    min_ball = round(target_plan_len * min_ratio_ball)
    if min_ball < 3:
        min_ball = 3

    max_ball = round(target_plan_len * max_ratio_ball)
    if max_ball <= min_ball:
        max_ball = min_ball + 1
    if max_ball > max_balls_id:
        max_ball = max_balls_id

    n_balls = random.randint(min_ball, max_ball)

    min_room = round(target_plan_len * min_ratio_room)
    max_room = round(target_plan_len * max_ratio_room)
    if min_room < 3:
        min_room = 3
    if max_room <= min_room:
        max_room = min_room + 1
    if max_room > max_rooms_id:
        max_room = max_rooms_id

    n_rooms = random.randint(min_room, max_room)

    return n_balls, n_rooms


def get_n_actions_contrasting(contrasting_param, version, target_plan_len):
    assert version == 'df' or version == 'nwf'

    min_contrasting_ratio = contrasting_param.get(f'min_{version}_actions', None)
    if min_contrasting_ratio is None:
        min_contrasting = 1
    else:
        min_contrasting = round(min_contrasting_ratio * target_plan_len)
        if min_contrasting == 0:
            min_contrasting = 1
    max_contrasting_ratio = contrasting_param.get(f'max_{version}_actions', 0.5)
    max_contrasting = round(max_contrasting_ratio * target_plan_len)
    if max_contrasting == 0:
        max_contrasting = 1

    n_contrasting_actions = random.randint(min_contrasting, max_contrasting)

    return n_contrasting_actions


