## Tabular Deep Learning Models Performance Evaluation

This part of the code is based on [TabSurvey](https://github.com/kathrinse/TabSurvey/tree/main) and extended on medical diagnosis tasks.

### Defining Datasets

`./config/` folder contains the `.yaml` files that include the necessary parameters for the experiments for each dataset. Some important points are:

- Setting `optimize_hyperparameters` to True conducts hyperparameter optimization with `n_trials` number of trials. If it is set to False, it uses the parameters defined in `best_params.yml`, which are the optimized parameters reported in the paper.

- `num_splits` sets the number of folders in cross-validation.

Any dataset can be added following this format, by defining the proper parameters inside the `.yml` file. Additionally, any necessary preprocessing statement and how to load the dataset itself should be added inside `./utils/load_data.py`.

### Experiments

See `testall.sh` to run the models over the datasets that have properly defined `.yml` config files.

