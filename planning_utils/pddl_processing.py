from typing import Union, Dict
from tarski.io import PDDLReader
from tarski.syntax import Atom, CompoundFormula


def process_problem_file(domain_path, problem_path):
    reader = PDDLReader(raise_on_error=True)
    reader.parse_domain(domain_path)
    problem = reader.parse_instance(problem_path)

    initial_state = [convert_pre2in(initial) for initial in list(problem.init.as_atoms())]
    goal_state = process_goal_conditions(problem)['pos_conditions']
    objects = get_object_types(problem).keys()
    object_type_dict = get_object_types(problem)

    return set(objects), set(initial_state), set(goal_state), problem.name, object_type_dict


def convert_pre2in(action: Union[str, Atom, CompoundFormula]):
    """
    Converts actions and predicates in the format that the PDDLReader outputs into the format that VAL
    expects
    i.e. clear(b) -> (clear b), pick-up(b) -> (pick-up b), stack(b, c) -> (stack b c)
    :param action:
    :return:
    """
    action = str(action)
    action_name, action_args = action.split('(')
    new_action_str = f'({action_name} {action_args}'
    new_action_str = new_action_str.replace(',', ' ')
    new_action_str = new_action_str.replace(' )', ')')
    return new_action_str


def get_object_types(problem) -> Dict[str, str]:

    object_type_dict = dict()

    problem_constants = list(problem.language.constants())

    for const in problem_constants:
        object_name = str(const.name)
        object_type = str(const.sort.name)
        object_type_dict[object_name] = object_type

    return object_type_dict


def process_goal_conditions(problem) -> Dict[str, list]:

    pos_goal_conditions = []
    neg_goal_conditions = []

    if isinstance(problem.goal, CompoundFormula):
        operator = problem.goal.connective
        if operator.name == 'And':
            for sub in problem.goal.subformulas:
                if isinstance(sub, Atom):
                    pos_goal_conditions.append(convert_pre2in(sub))
                elif sub.connective.name == 'Not':
                    assert len(sub.subformulas) == 1
                    pred_str = convert_pre2in(sub.subformulas[0])
                    neg_goal_conditions.append(pred_str)
        elif operator.name == 'Not':
            assert len(problem.goal.subformulas) == 1
            pred_str = convert_pre2in(problem.goal.subformulas[0])
            neg_goal_conditions.append(pred_str)
        else:
            raise ValueError

    elif isinstance(problem.goal, Atom):
        pos_goal_conditions.append(convert_pre2in(problem.goal))

    else:
        raise ValueError

    assert len(neg_goal_conditions) == 0
    goal_conditions = {'pos_conditions': pos_goal_conditions,
                       'neg_conditions': neg_goal_conditions}

    return goal_conditions
