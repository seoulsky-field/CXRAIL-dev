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


def raytune(hydra_cfg):
    assert (hydra_cfg.mode.execute_mode == 'raytune') , "change hydra mode into raytune. default mode should be executed in train.py"

    param_space = OmegaConf.to_container(instantiate(hydra_cfg.mode.param_space))
    tune_config = instantiate(hydra_cfg.mode.tune_config)
    run_config = instantiate(hydra_cfg.mode.run_config)
    
    # execute run
    tuner = tune.Tuner(
        trainable = tune.with_resources(partial(trainval, hydra_cfg=hydra_cfg), # 그냥 hydra_cfg넣으면 에러남
                                        {'cpu': int(round(multiprocessing.cpu_count()/2)), 
                                        'gpu': int(torch.cuda.device_count()),}),
        param_space = param_space,
        tune_config = tune_config,
        run_config = run_config
    )
    analysis = tuner.fit()


    # tuner 일때는 다름
    '''
    AttributeError: 'ResultGrid' object has no attribute 'get_best_trial'
    '''
    # best_trial = result.get_best_trial(metric ="loss", mode="min", scope="last")
    # print("Best trial config: {}".format(best_trial.config))
    # print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    # print("Best trial final validation accuracy: {}".format(best_trial.last_result["val_score"]))

    # model = instantiate(hydra_cfg.model)
    # model = model.to(device)
    #best_checkpoint_dir = best_trial.checkpoint.value
    # model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    # model.load_state_dict(model_state)

def default(hydra_cfg):
    config = hydra_cfg.mode
    #assert (config.execute_mode == 'default'), "change hydra mode into default. Ray should be executed in main.py"
    trainval(config, hydra_cfg)

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