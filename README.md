# Transformer Plan Verification

This is the repository for our ICML'26 paper, 
["On the ability of Transformers to verify plans"](https://arxiv.org/abs/2603.19954).

## Data Generation

We use the `data_generation` directory to generate the data for the experiments.

- Colors: see the `data_generation/color/README.md` file for more details.

- Lights out: `python ./data_generation/lights_out/run_generation.py --config [CONFIG]`
   * example configs (with the parameters used in our experiments) for generating data for the variant with conditional effects and without can be found in the ./configs/lights_out folder

- Grippers heavy: `python ./data_generation/grippers_heavy/run_generation.py --config [CONFIG]`
   * example configs (with the parameters used in our experiments) for generating data for the variant with conditional effects and without can be found in the ./configs/grippers_heavy foler

## Training

We use the `training` directory to train the models for the experiments.

We process data differently for each domain, each of which has a different folder. 
Within each domain folder, we also have a ``run.py`` file that generates the experiments (for HTCondor).


## Citation
```
@inproceedings{sarrof2026abilitytransformersverifyplans,
      title={On the Ability of Transformers to Verify Plans}, 
      author={Yash Sarrof and Yupei Du and Katharina Stein and Alexander Koller and Sylvie Thiébaux and Michael Hahn},
      booktitle={Forty-third International Conference on Machine Learning},
      year={2026}
}
```
