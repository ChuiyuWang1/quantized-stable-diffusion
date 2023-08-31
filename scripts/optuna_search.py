import optuna
from scripts.sample_eval import sampling_main, evaluation

def objective(trial):
    dataset = "celebahq"
    mode = "validation"
    resume = "models/ldm/celeba256/model.ckpt"
    log_dir = "samples/celeba256_search"
    quant_mode = "integer" # either "integer" or "block_fp"
    sample_dir, mem_density, total_flops_bitwidth = sampling_main(trial, quant_mode, resume, log_dir, n_samples=1000, custom_steps=100)
    metric = evaluation(sample_dir, dataset, mode)
    return [metric["frechet_inception_distance"], mem_density, total_flops_bitwidth]

# We use the multivariate TPE sampler.
sampler = optuna.samplers.TPESampler()

storage_name = "sqlite:///optuna.db"

study = optuna.create_study(sampler=sampler, direction=['minimize', 'minimize', 'minimize'],
                            study_name="celebahq_int_quant_search", storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=100)

best_params = study.best_params
best_value = study.best_value

# Save best results to a log file
with open("optuna_results.txt", "w") as f:
    f.write("Best parameters: {}\n".format(best_params))
    f.write("Best value: {}\n".format(best_value))

print("Best parameters:", best_params)
print("Best value:", best_value)