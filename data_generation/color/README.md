# Color Bags Data Generation

## Data Generation

We use the `run_generation.py` script to generate the data.
The key generator is in the `generator.py` file.

After generating the data, we use the `training/color/merge_direct_tokenized.py` script 
to merge the data into a single JSONL file and a split file.
