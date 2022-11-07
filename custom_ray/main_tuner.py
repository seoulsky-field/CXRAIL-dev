import os
import torch
import torch.nn as nn
from functools import partial
# ray
import ray
from ray import air, tune
from ray.tune import Trainable, run
from ray.tune.schedulers import ASHAScheduler
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import ScalingConfig
from omegaconf import DictConfig, OmegaConf

# reporter
from ray.tune import CLIReporter
from ray.tune.experiment import Trial
from typing import Any, Callable, Dict, List, Optional, Union
from utils.custom_reporter import TrialTerminationReporter

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
    param_space = {
        'lr': tune.loguniform(0.0001, 0.1),
        'batch_size': tune.choice([128, 256]),
    }
    tune_config = tune.TuneConfig(
        search_alg = HyperOptSearch(space=param_space,metric="loss",mode="min"),
        # scheduler = ASHAScheduler(metric="loss", mode="min"),
        num_samples=10
    )
    reporter = TrialTerminationReporter(       
        parameter_columns=["lr"],
        metric_columns=["loss", "val_loss", "val_score",  "current_epoch",  "progress_of_epoch" ])
    
    run_config = air.RunConfig(
        progress_reporter=reporter,
        local_dir="./ray_logs/",
        name="test_experiment",
        verbose=2, # 0 silent, 1 = only status updates, 2 = status and brief results, 3 = status and detailed results. Defaults to 2.
        )
    
    tuner = tune.Tuner(
        trainable = tune.with_resources(
            partial(trainval, hydra_cfg=cfg), 
            {
                'cpu': 8, 
                'gpu': int(torch.cuda.device_count()),
                }
            ),
        #param_space = param_space,
        tune_config = tune_config,
        run_config = run_config
    )
    result = tuner.fit()


    best_trial = result.get_best_trial(metric ="loss", mode="min", scope="last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

    model = instantiate(cfg.model)
    model = model.to(device)
    # model = instantiate(cfg.model)
    # model = model.to(device)
    # best_checkpoint_dir = best_trial.checkpoint.value
    # model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    # model.load_state_dict(model_state)


if __name__ == "__main__":
    main()