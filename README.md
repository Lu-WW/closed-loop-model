## File contents
- `base.py` contains the basic parameters and functions for the model.
You can also find the configuration of some basic alternative models or simulation settings in this file.

- `alter.py` contains the functions that was used in alternative models or simulation settings.

- `experiment.py` contains the class to run the experiment (with default setting).
  
- `analyzer.py` contains the class to analyze the results of experiments.


- `utils.py` contains some useful functions for the model.

- `run_all.py` is the main file to run all experiments.

- `GLM.py` contains the code to run generalized linear model (GLM) to estimate the effect of stimulus and RT on model accuracy and metacognition.


- `baseline.py`, `reverse.py`, `pulse.py`,` pulse_para.py`, `attention.py`, `fixed_duration_experiment.py` provide the codes to run and analyze corresponding simulation settings.

- `compare.py` provide some useful functions to compare the results of different simulations.

- `aDDM.py` provide simple code to simulate the aDDM model.


- `psych_kernel.py` contains the code to calculate psychophysical kernel and compare across settings.

## Basic usage

Use `-s` to specify running a certain setting, use `-m` for models.

See `base.py` or other files correspond to simulation settings for supported settings and models.

See `run_all.py` for examples.


For example, following command will run the default model with noise level 1:


```shell
python experiment.py -s noise_1 
```

use `-e` to evaluate and analyze stored results
