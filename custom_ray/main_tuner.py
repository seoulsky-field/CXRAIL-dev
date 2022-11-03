import os
import torch
import torch.nn as nn

# ray
from ray import air, tune
from ray.tune import Trainable, run
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import ScalingConfig

# hydra
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from hydra.utils import instantiate

from custom_train import trainval

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@hydra.main(
    version_base = None, 
    config_path='config', 
    config_name = 'config'
    )
def set_parameters(cfg: DictConfig):
    # trainable
    trainable = trainval(cfg, search_space)

    # param_space
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        #"batch_size": tune.choice([128, 256, 512])
    }

    # tune_config
    tune_config = tune.TuneConfig(
        search_alg = HyperOptSearch(space=search_space,metric="mean_accuracy",mode="max"),
        scheduler = ASHAScheduler(metric="mean_accuracy", mode="max"),
        num_samples=10
    )
    # run_config
    reporter = CLIReporter(
        parameter_columns=["lr"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    run_config = air.RunConfig(progress_reporter=reporter)
    # run_config=air.RunConfig(
    #         local_dir="/home/CheXpert_code/jieon/Baseline/raytune_class3/saved/logs",
    #         name="test_experiment"),

    return trainable, search_space, tune_config, run_config 


def main(cfg: DictConfig):
    trainable, search_space, tune_config, run_config = set_parameters()
    
    tuner = tune.Tuner(
        trainable = trainable,
        param_space = search_space,
        tune_config = tune_config,
        run_config = run_config
    )
    result = tuner.fit()


    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.search_space))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

    model = instantiate(cfg.model)
    model = model.to(device)
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    model.load_state_dict(model_state)


if __name__ == "__main__":
    main()