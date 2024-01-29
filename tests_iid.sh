#!/bin/bash

# Define arrays of flag values

learning_rate=(0.001 0.01  0.05 0.1)
batch_size=(32 64 128 256)
decay=(0 1e-2 1e-3 1e-5)
rounds=(5 10 15 20 30 50)
n_clients=(5 10 15 20)
epochs=(1 5 10 15)		
seed=(1234567890 1472583690)

# Output directory for results


# Create the output directory if it doesn't exist
#mkdir -p "$output_dir"

# Loop through combinations of flags
for param1 in "${learning_rate[@]}"; do
  for param2 in "${batch_size[@]}"; do
    for param3 in "${decay[@]}"; do
      for param4 in "${rounds[@]}"; do
        for param5 in "${n_clients[@]}"; do
          for param6 in "${epochs[@]}"; do
            for param7 in "${seed[@]}"; do

              # Build the command with the current combination of flags
              command="python main.py --dataset femnist --model cnn --federated\
                --lr=$param1 --bs=$param2 --wd=$param3 --num_rounds=$param4\
                --clients_per_round=$param5 --num_epochs=$param6 --seed=$param7"
              
               identifier="${param1}_${param2}_${param3}_${param4}_${param5}_${param6}_${param7}"

              $command
              
              mv *.csv ./Results/$identifier.csv
		
            done
          done
        done
      done
    done
  done
done

echo "Simulation complete."

