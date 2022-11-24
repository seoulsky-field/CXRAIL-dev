import os
import torch
import torch.nn as nn
import multiprocessing
from functools import partial

# ray
import ray
from ray import air, tune
from ray.tune import Trainable, run
from ray.tune.schedulers import ASHAScheduler
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import ScalingConfig

# reporter
from ray.tune import CLIReporter
from ray.tune.experiment import Trial
from typing import Any, Callable, Dict, List, Optional, Union
from custom_utils.custom_reporter import TrialTerminationReporter

# hydra
import hydra
from omegaconf import DictConfig, OmegaConf, errors
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
# 내부모듈
from train import trainval

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def default(hydra_cfg):
    config = hydra_cfg.mode
    trainval(config, hydra_cfg)


def raytune(hydra_cfg):
    param_space = OmegaConf.to_container(instantiate(hydra_cfg.mode.param_space))
    scheduler = instantiate(hydra_cfg.mode.scheduler)
    search_alg = instantiate(hydra_cfg.mode.search_alg, space=param_space)
 
    reporter = TrialTerminationReporter(
        parameter_columns = param_space.keys(),
        metric_columns= ['epoch', 'Batch_ID', 'loss', 'val_loss', 'val_score', 'best_val_score', 'progress_of_epoch'])
    scheduler = instantiate(hydra_cfg.mode.scheduler)

    # execute run
    result = tune.run(
        partial(trainval, hydra_cfg = hydra_cfg),
        config = param_space,
        num_samples = hydra_cfg.mode.num_samples,
        scheduler = scheduler,
        search_alg = search_alg, 
        progress_reporter=reporter,
        resources_per_trial={
                'cpu': int(round(multiprocessing.cpu_count()/2)), 
                'gpu': int(torch.cuda.device_count()),
                },
        local_dir = HydraConfig.get().sweep.dir,
        name = HydraConfig.get().sweep.subdir)


    best_trial = result.get_best_trial(metric ="loss", mode="min", scope="last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["val_score"]))

    # model = instantiate(hydra_cfg.model)
    # model = model.to(device)
    #best_checkpoint_dir = best_trial.checkpoint.value
    # model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    # model.load_state_dict(model_state)


@hydra.main(
    version_base = None, 
    config_path='config', 
    config_name = 'config'
)
def main(hydra_cfg: DictConfig):
    if hydra_cfg.mode.execute_mode == 'default':
        print("mode=defalt")
        default(hydra_cfg)
    
    elif hydra_cfg.mode.execute_mode =='raytune':
        print("mode=raytune")
        raytune(hydra_cfg)


if __name__ == "__main__":
    main()


