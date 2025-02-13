#!/bin/bash
# run_experiments.sh
# This script runs experiments on main.py with:
#   - quantization = 4
#   - max_rounds = 50
#   - structure_id: C1, C2, C4, ..., C21 (skipping C3 since it's the one shot example)
#   - modality flags: use_img only, both use_img and use_json, and use_json only
#   - shot type: oneshot and zeroshot

# Set constant parameters
QUANTIZATION=4
MAX_ROUNDS=50

# Create an array of structure IDs from C1 to C21, skipping C3.
structures=()
for i in $(seq 1 21); do
  if [ "$i" -eq 3 ]; then
    continue
  fi
  structures+=("C$i")
done

# Define modality combinations.
# We'll use internal names to later decide which flags to pass.
modalities=("use_img" "use_img_and_json" "use_json")

# Define shot types.
shots=("oneshot" "zeroshot")

# Loop over each combination of structure, modality, and shot.
for structure in "${structures[@]}"; do
  for modality in "${modalities[@]}"; do
    for shot in "${shots[@]}"; do
      
      # Initialize an empty string for additional flags.
      flags=""
      
      # Set modality flags.
      if [ "$modality" == "use_img" ]; then
        flags+=" --use_img"
      elif [ "$modality" == "use_img_and_json" ]; then
        flags+=" --use_img --use_json"
      elif [ "$modality" == "use_json" ]; then
        flags+=" --use_json"
      fi
      
      # Set shot flag.
      if [ "$shot" == "oneshot" ]; then
        flags+=" --oneshot"
      elif [ "$shot" == "zeroshot" ]; then
        flags+=" --zeroshot"
      fi
      
      # Echo the current configuration (optional).
      #echo "Running experiment with structure_id: ${structure}, modality: ${modality}, shot: ${shot}"
      
      # Execute the experiment.
      python3 src/main.py --quantization ${QUANTIZATION} --max_rounds ${MAX_ROUNDS} --structure_id ${structure} ${flags}
      #echo "python main.py --quantization ${QUANTIZATION} --max_rounds ${MAX_ROUNDS} --structure_id ${structure} ${flags}"
      
    done
  done
done
