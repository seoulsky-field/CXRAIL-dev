import os
import torch
import torch.nn as nn
from functools import partial

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

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

@hydra.main(
    version_base = None, 
    config_path='config', 
    config_name = 'config'
    )

def main(cfg: DictConfig):
    param_space = {
        'lr': tune.loguniform(0.0001, 0.1),
        'batch_size': tune.choice([128, 256]),
    }
    tune_config = tune.TuneConfig(
        search_alg = HyperOptSearch(space=param_space,metric="loss",mode="min"),
        scheduler = ASHAScheduler(metric="loss", mode="min"),
        num_samples=10
    )
    reporter = CLIReporter(
        parameter_columns=["lr"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    run_config = air.RunConfig(progress_reporter=reporter,
        local_dir="./ray_logs/",
        name="test_experiment")
    
    tuner = tune.Tuner(
        trainable = partial(trainval, hydra_cfg=cfg),
        #param_space = param_space,
        tune_config = tune_config,
        run_config = run_config
    )
    result = tuner.fit()


    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.cfg.ray))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

    # model = instantiate(cfg.model)
    # model = model.to(device)
    # best_checkpoint_dir = best_trial.checkpoint.value
    # model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    # model.load_state_dict(model_state)


if __name__ == "__main__":
    main()