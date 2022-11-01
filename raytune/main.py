#ray
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import ScalingConfig

import os
import wandb
import torch
from ray.tune.logger import DEFAULT_LOGGERS
# from ray.tune.integration.wandb import WandbLogger
from pytorch_lightning.loggers import WandbLogger

#in
from config import cfg , seed_everything
from models import select_model
from dataset import get_dataloader
from train import train_chexpert

import argparse
import warnings

warnings.filterwarnings(action="ignore")
seed_everything(cfg.seed)

parser = argparse.ArgumentParser(description="Chexpert pipeline")
parser.add_argument("--model", default="efficientnet", type=str)
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--sweep", action="store_true")
parser.add_argument("--val_every", default=1, type=int)

args = parser.parse_args()
#-------------------------------------------------------------------------#

def main():
    model = select_model(args.model)
    num_samples=10, 
    max_num_epochs=10


    ray_config = {
        "lr": hp.loguniform("lr", 1e-10, 0.1),
        "batch_size": tune.choice([2, 4, 8, 16]),
        "wandb": {
            "project": "Optimization_Project",
            "api_key_file": "/path/to/file",
            "log_config": True
        }
    }
    hyperopt_search = HyperOptSearch(ray_config, metric="mean_accuracy", mode="max")

    trainable = train_chexpert(
        args=args,
        ray_config = ray_config,
        model=model,
        device=cfg.device,
    )

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
 

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    
    
    # tune 실행 #1 (tune.run)  
    result = tune.run(
        trainable,
        #resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=cfg,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        search_alg= hyperopt_search,
        loggers=DEFAULT_LOGGERS + (WandbLogger, )
    )

    
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    model.load_state_dict(model_state)

    # test_acc = test_accuracy(model, device)
    # print("Best trial test set accuracy: {}".format(test_acc))


# tune 실행 #2. (tune.Tuner)

# # import run
# model = select_model(args.model)
# trainable = train_chexpert(
#     args=args,
#     device=cfg.device,
# )

# # set search space

# search_space = {
#     "lr": hp.loguniform("lr", 1e-5, 1e-1),
#     "momentum": hp.uniform("momentum", 0.1, 0.9),
#     #"weight_decay"
#     "batch_size": tune.choice([2, 4, 8, 16])
# }


# def chexpert_ray(trainable):
#     tuner = tune.Tuner(
#         trainable = trainable,
#         param_space = search_space,
#         run_config=air.RunConfig(
#             local_dir="/home/CheXpert_code/jieon/Baseline/Dev_RayTune/saved/logs",
#             name="test_experiment"),
#         tune_config=tune.TuneConfig(
#             num_samples=20,
#             scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
#             search_alg=hyperopt_search,
#         ),
#     )
#     results = tuner.fit()
#     return results

# def with_wandb(trainable):
#     tune.run(
#         trainable,
#         config={
#             # define search space here
#             "parameter_1": tune.choice([1, 2, 3]),
#             "parameter_2": tune.choice([4, 5, 6]),
#             # wandb configuration
#             "wandb": {
#                 "project": "Optimization_Project",
#                 "api_key_file": "/path/to/file",
#                 "log_config": True
#             }
#         },
#         loggers=DEFAULT_LOGGERS + (WandbLogger, ))


if __name__ == "__main__":
    # args = parser.parse_args()
    # if args.sweep:
    #     sweep_id = wandb.sweep(sweep_config, entity="jieonh", project="Chexpert-view-classification")
    #     wandb.agent(sweep_id, main, count=20)
    # else:
    #     main()
    main()