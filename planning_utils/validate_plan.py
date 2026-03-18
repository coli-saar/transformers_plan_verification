import os
import random
from planning_utils.paths import VAL, TEMP_DIR


def run_validation(domain_file, problem_file, plan_file, plan_end_tag: bool = False):

    rem_temp_file = False
    if plan_end_tag:
        # Remove the PLAN_END tag
        orig_plan = []

        with open(plan_file, 'r') as f:
            for action_line in f.readlines():
                action = action_line.strip()
                if action.startswith(';'):
                    continue
                if action == '(PLAN_END)':
                    continue
                orig_plan.append(action.strip())

        rand_id = random.randint(0, 10000000)
        temp_plan_file = TEMP_DIR / f'temp_inst_{rand_id}_plan.txt'
        with open(temp_plan_file, 'w') as f:
            plan_str = '\n'.join(orig_plan)
            f.write(plan_str)
        rem_temp_file = True
    else:
        temp_plan_file = plan_file

    val = VAL

    if not os.path.exists(domain_file):
        raise FileNotFoundError(f'domain file does not exist: {domain_file}')
    if not os.path.exists(problem_file):
        raise FileNotFoundError(f'problem file does not exist: {problem_file}')
    if not os.path.exists(temp_plan_file):
        raise FileNotFoundError(f'plan files does not exit: {temp_plan_file}')

    cmd = f'{val}/validate -v {domain_file} {problem_file} {temp_plan_file}'

    val_response = os.popen(cmd).read()
    if 'No such file or directory' in val_response:
        print('Could not find file')
        print(val_response)
        raise FileNotFoundError

    reached_goal, executable = parse_val_output(response=val_response)
    #print(val_response)
    if rem_temp_file:
        os.remove(temp_plan_file)

    return reached_goal, executable


def parse_val_output(response: str) -> (bool, bool):
    """
    :param response:
    :return:
    """
    goal_satisfied = False
    plan_executable = False

    reached_execution = False
    reached_unmet_pre = False
    reached_effect = False
    reached_unmet_pre_at_some_time = False

    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if 'Successful plans:' in line or 'Failed plans:' in line:
            break

        if reached_execution:
            if 'Plan failed because' in line:
                reached_unmet_pre = True
                reached_unmet_pre_at_some_time = True
                plan_executable = False
            else:                               # then the plan is executable
                reached_effect = True

        if reached_effect and line:
            if 'executed successfully' in line:
                plan_executable = True
            elif 'Plan valid' in line:
                goal_satisfied = True   # plan is valid if plan is executable and goal is satisfied

        elif reached_effect and not line:
            reached_effect = False

        if reached_unmet_pre and not line:    # processed all unmet preconditions
            reached_unmet_pre = False

        if line.startswith('Checking next happening'):
            reached_execution = True

        if 'Bad plan description!' in line:
            raise ValueError('Bad plan description')

        if 'Bad problem file!' in line:
            raise ValueError('Bad problem file: probably path is missing')

        if 'Bad operator' in line:
            raise ValueError('Bad operator in plan')

    # Make sure that plan fails for a relevant reason
    if not plan_executable:
        if not reached_unmet_pre_at_some_time:
            print('Response:')
            print(response)
        assert reached_unmet_pre_at_some_time

    return goal_satisfied, plan_executable

