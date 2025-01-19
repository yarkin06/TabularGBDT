## Traditional ML and GBDT Models Performance Evaluation

Experiments can be run using `./local_launch.py`, where you define the number of folds for the cross-validation and choose the datasets you run the models over, where the models are defined in `./main.py` file. The code has `data_prep` or `train` modes, in which you first prepare and save the data, then conduct the training. 

Dataset loading and preprocessing is conducted inside `./utils.py`. Any desired dataset can be added in `./local_launch.py` with the proper preprocessing defined inside `./utils.py`.