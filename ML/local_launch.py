import os

# mode = 'data_prep'
mode = 'train'

folds = 8

datasets = [
        'Cardiovascular-Disease-dataset',  
        'heart_failure', 
        'parkinsons', 
        'eeg-eye-state',
        'eye_movements',
        'arcene',
        'Prostate',
    ]

for dataset in datasets:
    file = f"--mode={mode} " \
            f"--data={dataset} " \
            f"--folds={folds}"
    
    cmd = f"python main.py {file}"
    os.system(cmd)