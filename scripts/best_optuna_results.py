import optuna
import json
import numpy as np

# Create and run the original study
storage_name = "sqlite:///optuna.db"
study = optuna.load_study(study_name="celebahq_int_quant_search", storage=storage_name)

# Retrieve all trials
trials = study.get_trials(deepcopy=False)

# Filter out trials with no value
valid_trials = [trial for trial in trials if trial.value is not None]

# Calculate the number of trials for the top 10%
top_n = int(len(valid_trials) * 0.1)

# Sort the trials by their values
sorted_trials = sorted(trials, key=lambda trial: trial.value if trial.value is not None else float('inf'))

# Initialize a dictionary to store the parameter distributions
parameter_distribution = {}

# Process the top 10% of trials
for trial in sorted_trials[:top_n]:
    params = trial.params
    for param_name, param_value in params.items():
        if param_name not in parameter_distribution:
            parameter_distribution[param_name] = {}
        if param_value not in parameter_distribution[param_name]:
            parameter_distribution[param_name][param_value] = 1
        else:
            parameter_distribution[param_name][param_value] += 1

# Print the parameter distribution
output_filename = "parameter_distribution.json"
with open(output_filename, 'w') as outfile:
    json.dump(parameter_distribution, outfile)

# Alternatively, you can save the parameter distribution into a dictionary for further use
# Example: distribution_dict[param_name] = distribution