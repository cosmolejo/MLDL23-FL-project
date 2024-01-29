#!/bin/bash

# Define arrays of flag values

learning_rate=(0.001 0.01 )
batch_size=(32 64 128)
decay=(0 1e-2)
epochs=(5 10 15 20 30 50)		
seed=(1234567890 42)

# Output directory for results


# Create the output directory if it doesn't exist
#mkdir -p "$output_dir"

# Loop through combinations of flags
for param1 in "${learning_rate[@]}"; do
  for param2 in "${batch_size[@]}"; do
    for param3 in "${decay[@]}"; do
      for param4 in "${epochs[@]}"; do
        for param5 in "${seed[@]}"; do

              # Build the command with the current combination of flags
              command="python main.py --dataset femnist --model cnn \
                --lr=$param1 --bs=$param2 --wd=$param3 --num_epochs=$param4 --seed=$param5"
              
               #identifier="${param1}_${param2}_${param3}_${param4}_${param5}_${param6}_${param7}"

              $command
              
              mv *.csv ./Results/. #$identifier.csv
		
        done
      done
    done
  done
done

echo "Simulation complete."

