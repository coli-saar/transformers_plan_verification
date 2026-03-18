import os
import json
import random
from argparse import ArgumentParser
from data_generation.lights_out.generator_original import generate_dataset_conditional
from data_generation.lights_out.generator_exponential import generate_dataset_exponential
from data_generation.lights_out.lights_out_exponential_helpers import generate_domain_file_exp
from planning_utils.paths import create_temp_dir


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config_file = args.config
    with open(config_file, 'r') as f:
        parameters = json.load(f)

    create_temp_dir()

    if parameters.get('next_inst_id', None) is None:
        parameters['next_inst_id'] = random.randint(0, 10000000)

    min_len = parameters['min_len']
    max_len = parameters['max_len']
    parameters['output_file'] = parameters['output_file'].replace('.jsonl', f'_{min_len}_{max_len}.jsonl')

    conditional_variant = parameters['conditional']
    del parameters['conditional']

    if not conditional_variant:
        if not os.path.exists(parameters['domain_file']):
            x_dim = parameters.get('x_dim', 5)
            y_dim = parameters.get('y_dim', 5)
            generate_domain_file_exp(x_dim=x_dim, y_dim=y_dim, out_file=parameters['domain_file'], write=True)

    output_file_path = os.path.join(parameters['data_dir'], parameters["output_file"])
    if os.path.exists(output_file_path):
        handle_existing = parameters.get('handle_existing', None)
        if handle_existing is None:
            error_message = f'The output file {output_file_path} already exists. Specify how to deal with existing files by adding the "handle_existing" parameter to the config file.\nSet it to "overwrite" for overwriting or "append" for continuing with the existing file'
            raise FileExistsError(error_message)
        else:
            del parameters['handle_existing']
            if handle_existing == 'overwrite':
                with open(output_file_path, 'w') as f:
                    f.write('')
            else:
                assert handle_existing == 'append'

    if conditional_variant:
        generate_dataset_conditional(** parameters)
    else:
        generate_dataset_exponential(**parameters)
