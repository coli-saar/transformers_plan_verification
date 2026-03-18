import os
import json
import random
from argparse import ArgumentParser
from data_generation.grippers_heavy.generator import generate_dataset
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

    parameters['well_formed'] = False
    parameters['delete_relaxed'] = False
    if parameters['version'] == 'well_formed':
        parameters['well_formed'] = True
    elif parameters['version'] == 'delete_free':
        parameters['delete_relaxed'] = True
    del parameters['version']

    output_file_path = os.path.join(parameters['dataset_dir'], parameters["output_file"])
    handle_existing = parameters.get('handle_existing', None)
    if os.path.exists(output_file_path):

        if handle_existing is None:
            error_message = f'The output file {output_file_path} already exists. Specify how to deal with existing ' \
                            f'files by adding the "handle_existing" parameter to the config file.\nSet it to ' \
                            f'"overwrite" for overwriting or "append" for continuing with the existing file'
            raise FileExistsError(error_message)
        else:
            if handle_existing == 'overwrite':
                with open(output_file_path, 'w') as f:
                    f.write('')
            else:
                assert handle_existing == 'append'
    if handle_existing is not None:
        del parameters['handle_existing']

    generate_dataset(**parameters)

