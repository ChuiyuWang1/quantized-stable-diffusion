import optuna
from scripts.sample_eval import sampling_main, evaluation

def objective(trial):
    dataset = "celebahq"
    mode = "validation"
    resume = "models/ldm/celeba256/model.ckpt"
    log_dir = "samples/celeba256_search"
    sample_dir = sampling_main(trial, resume, log_dir, n_samples=1000, custom_steps=100)
    metric = evaluation(sample_dir, dataset, mode)
    return metric["frechet_inception_distance"]

# We use the multivariate TPE sampler.
sampler = optuna.samplers.TPESampler(multivariate=True)


study = optuna.create_study(sampler=sampler, direction='minimize')
study.optimize(objective, n_trials=100)