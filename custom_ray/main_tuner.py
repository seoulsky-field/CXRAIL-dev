import os
import torch
import torch.nn as nn
import multiprocessing
from functools import partial

# ray
from ray import air, tune
from ray.tune import Trainable, run
from ray.tune import CLIReporter
#from ray.tune.schedulers import ASHAScheduler
#from hyperopt import hp
#from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import ScalingConfig

# reporter
from ray.tune import CLIReporter
from ray.tune.experiment import Trial
from typing import Any, Callable, Dict, List, Optional, Union
from utils.custom_reporter import TrialTerminationReporter

# hydra
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


from custom_train import trainval

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@hydra.main(
    version_base = None, 
    config_path='config', 
    config_name = 'config'
    )

def main(cfg: DictConfig):

    param_space = OmegaConf.to_container(instantiate(cfg.ray.param_space))
    tune_config = tune.TuneConfig(
        search_alg = instantiate(cfg.ray.search_alg, space=param_space),
        num_samples = cfg.ray.num_samples
    )
    reporter = instantiate(cfg.ray.reporter)
    run_config = instantiate(cfg.ray.run_config, progress_reporter = reporter)
    
    tuner = tune.Tuner(
        trainable = tune.with_resources(
            partial(trainval, hydra_cfg=cfg), 
            {
                'cpu': int(round(multiprocessing.cpu_count()/2)), 
                'gpu': int(torch.cuda.device_count()),
                }
            ),
        param_space = param_space,
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