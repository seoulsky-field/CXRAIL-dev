from multiprocessing import cpu_count
from functools import partial

from ray import tune
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from ray import air

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

import timm

from custom_train import trainval

@hydra.main(
    version_base = None, 
    config_path='../hydra_ref/config', 
    config_name = 'basic_config'
    )
def main(cfg: DictConfig):
    param_space = {
        'lr': tune.loguniform(0.0001, 0.1),
        'batch_size': tune.choice([32, 64]),
    }

    scheduler = ASHAScheduler(
        metric='loss',
        mode='min',
    )

    reporter = tune.CLIReporter(
        metric_columns=['loss', 'val_loss']
    )

    result = tune.run(
        partial(trainval, hydra_cfg=cfg),
        config=param_space,
        num_samples=cfg.tuning_iteration,
        scheduler=scheduler,
        progress_reporter=reporter,
        resources_per_trial={"cpu": int(round(cpu_count()/2)), "gpu": 1},
    )

if __name__=="__main__":
    main()