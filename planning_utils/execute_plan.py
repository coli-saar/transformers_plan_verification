import pymimir as mm


def execute_plan_mimir(domain_file, problem_file, plan):

    parser = mm.PDDLParser(str(domain_file), str(problem_file))
    problem = parser.get_problem()
    factories = parser.get_pddl_factories()

    successor_generator = mm.LiftedApplicableActionGenerator(problem, factories)
    state_repository = mm.StateRepository(successor_generator)
    current_state = state_repository.get_or_create_initial_state()

    all_applicable_actions_mimir = successor_generator.compute_applicable_actions(current_state)
    all_appl_actions_str = [action.to_string_for_plan(factories) for action in all_applicable_actions_mimir]
    action_mappings = {ac_str: ac_mm for (ac_str, ac_mm) in zip(all_appl_actions_str, all_applicable_actions_mimir)}

    for action_str in plan:
        current_state = state_repository.get_or_create_successor_state(
            current_state, action_mappings[action_str]
        )

        all_applicable_actions_mimir = successor_generator.compute_applicable_actions(current_state)
        all_appl_actions_str = [action.to_string_for_plan(factories) for action in all_applicable_actions_mimir]

        action_mappings = {ac_str: ac_mm for (ac_str, ac_mm) in zip(all_appl_actions_str, all_applicable_actions_mimir)}

    return current_state, all_appl_actions_str
