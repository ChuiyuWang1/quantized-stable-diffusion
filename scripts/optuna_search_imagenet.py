import optuna
from scripts.sample_eval import sampling_main, evaluation

def objective(trial):
    dataset = "imagenet" # celebahq or imagenet
    mode = "validation" # train or validation
    resume = "models/ldm/cin256-v2/model.ckpt" # "models/ldm/celeba256/model.ckpt" or "models/ldm/cin256-v2/model.ckpt"
    log_dir = "samples/cin256-v2_search" # "samples/celeba256_search" or "samples/cin256-v2_search"
    quant_mode = "integer" # either "integer" or "block_fp"
    sample_dir, mem_density, avg_flops_bitwidth = sampling_main(trial, quant_mode, resume, log_dir, n_samples=150, custom_steps=100)
    metric = evaluation(sample_dir, dataset, mode)
    return [metric["frechet_inception_distance"]/20.0, 32 / mem_density, avg_flops_bitwidth ** 0.5]

# We use the multivariate TPE sampler.
sampler = optuna.samplers.TPESampler()

storage_name = "sqlite:///optuna_final.db"

study = optuna.create_study(sampler=sampler, directions=['minimize', 'minimize', 'minimize'],
                            study_name="cin_int_quant_search", storage=storage_name, load_if_exists=True)
with open("optuna_start_imagenet.json", "r") as json_file:
    loaded_params = json.load(json_file)
study.enqueue_trial(loaded_params, skip_if_exists=True)
study.optimize(objective, n_trials=100)

best_params = study.best_params
best_value = study.best_value

# Save best results to a log file
with open("optuna_results.txt", "w") as f:
    f.write("Best parameters: {}\n".format(best_params))
    f.write("Best value: {}\n".format(best_value))

print("Best parameters:", best_params)
print("Best value:", best_value)