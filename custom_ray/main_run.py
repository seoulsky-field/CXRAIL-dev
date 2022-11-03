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
def main(cfg: DictConfig):

    # set parameters
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        #"batch_size": tune.choice([128, 256, 512])
    }
    trainable = trainval(cfg, search_space)
    search_alg = HyperOptSearch(space=search_space,metric="mean_accuracy",mode="max"),
    scheduler = ASHAScheduler(metric="mean_accuracy", mode="max"),
    reporter = CLIReporter(
        parameter_columns=["lr"],
        metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        trainable = trainable,
        config = search_space,
        num_samples=10,
        scheduler = scheduler,
        search_alg = search_alg, 
        progress_reporter=reporter)


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
    # You can change the number of GPUs per trial here:
    main()