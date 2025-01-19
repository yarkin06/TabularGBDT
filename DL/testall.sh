#!/bin/bash

N_TRIALS=36
EPOCHS=1000

TORCH_ENV="TabularGBDT"

declare -A MODELS
MODELS=( 
         ["MLP"]=$TORCH_ENV
         ["TabNet"]=$TORCH_ENV
         ["VIME"]=$TORCH_ENV
         ["TabTransformer"]=$TORCH_ENV
         ["STG"]=$TORCH_ENV
          )

CONFIGS=( 
          "config/arcene.yml"
          "config/Cardiovascular-Disease-dataset.yml"
          "config/eeg-eye-state.yml"
          "config/eye_movements.yml"
          "config/heart_failure.yml"
          "config/parkinsons.yml"
          "config/Prostate.yml"
          )

# conda init bash
eval "$(conda shell.bash hook)"

for config in "${CONFIGS[@]}"; do

  for model in "${!MODELS[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s with %s in env %s\n\n' "$model" "$config" "${MODELS[$model]}"

    conda activate "${MODELS[$model]}"

    python train.py --config "$config" --model_name "$model" --n_trials $N_TRIALS --epochs $EPOCHS

    conda deactivate

  done

done
