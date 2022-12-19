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
from ray.tune import CLIReporter
from ray.tune.experiment import Trial
from ray.air import session
from typing import Any, Callable, Dict, List, Optional, Union
from custom_utils.custom_reporter import TrialTerminationReporter

# hydra
import hydra
from omegaconf import DictConfig, OmegaConf, errors
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

# optuna
import optuna
from optuna.trial import TrialState

#WandB
import wandb
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.tune.integration.wandb import (WandbTrainableMixin, wandb_mixin)

# 내부모듈
from train import trainval

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
working_dir = os.getcwd()

def optuna_sweep(trial:optuna.trial.Trial, hydra_cfg):
    # print("start optuna sweep function")
    # hydra_cfg.hparams_search.params.get('trial', trial)
    # config = hydra_cfg.hparams_search.params
    # print(type(config))
    # print('-'*100)
    # print(config)
    config = {'lr': trial.suggest_float('lr', 0.00001, 0.1),
            'weight_decay': trial.suggest_float('weight_decay', 0.00001, 0.1),
            'rotate_degree':trial.suggest_loguniform('rotate_degree', 0.1, 15),
            'batch_size': trial.suggest_categorical('batch_size',[64, 128])}
    print('-'*100)
    print(config)
    result_metrics, epoch = trainval(config, hydra_cfg, best_val_roc_auc = 0)
    print("RESLT_METRICS: ", result_metrics)
    trial.report(result_metrics, epoch)

    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

def ray_tune(config, hydra_cfg):
    result_metrics, epoch = trainval(config, hydra_cfg, best_val_roc_auc = 0)
    session.report(result_metrics, epoch)


# def wandb_sweep(hydra_cfg):
#     run = wandb.init(project="CXRAIL")

#     config = {
#         "learning_rate": 1e-3,
#         "epochs": 5,
#         "batch_size": 64
#     }
#     wandb.config.update(cfg)
#     result_metrics, epoch = trainval(config, hydra_cfg, best_val_roc_auc = 0)
#     wandb.log({"train_loss": total_loss / len(dataloader)}, step=epoch)

@hydra.main(
    version_base = None, 
    config_path='configs', 
    config_name = 'config.yaml'
)
def main(hydra_cfg:DictConfig):
    print('working dir: ' + os.getcwd())

    # search
    if hydra_cfg.get('hparams_search', None):
        name = hydra_cfg.hparams_search.name
        print("NAME:", name)

        if  name == 'optuna':
            study = optuna.create_study(direction="maximize")
            study.optimize(partial(optuna_sweep, hydra_cfg=hydra_cfg), n_trials=10, timeout=600)
            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            # print("Study statistics: ")
            # print("  Number of finished trials: ", len(study.trials))
            # print("  Number of pruned trials: ", len(pruned_trials))
            # print("  Number of complete trials: ", len(complete_trials))
            # print("Best trial:")
            # trial = study.best_trial

            # print("  Value: ", trial.value)
            # print("  Params: ")
            # for key, value in trial.params.items():
            #     print("    {}: {}".format(key, value))


        elif name == 'raytune':
            param_space = OmegaConf.to_container(instantiate(hydra_cfg.hparams_search.param_space))
            tune_config = instantiate(hydra_cfg.hparams_search.tune_config)
            run_config = instantiate(hydra_cfg.hparams_search.run_config, callbacks=[WandbLoggerCallback(api_key=hydra_cfg.api_key, project="Wandb_ray_lrcontrol")])
            
            # execute run
            tuner = tune.Tuner(
                trainable = tune.with_resources(partial(ray_tune, hydra_cfg=hydra_cfg),
                                                {'cpu': int(round(multiprocessing.cpu_count()/2)), 
                                                'gpu': int(torch.cuda.device_count()),}),
                param_space = param_space,
                tune_config = tune_config,
                run_config = run_config
            )
            analysis = tuner.fit()

        # elif name == 'wandb_sweep':
        #     config = hydra_cfg.hparams_search
        
    # search X
    else:
        print('default')
        config = hydra_cfg
        trainval(config, hydra_cfg, best_val_roc_auc = 0)

    os.chdir(working_dir)


    

if __name__ == "__main__":
    main()
    