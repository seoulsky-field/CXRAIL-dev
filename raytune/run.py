#ray
from ray import air, tune
from ray.tune import Trainable, run

from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import ScalingConfig
from functools import partial
import os
import wandb
import torch
from ray.tune.logger import DEFAULT_LOGGERS
# from ray.tune.integration.wandb import WandbLogger
from pytorch_lightning.loggers import WandbLogger

#in
from config import cfg , seed_everything
from train import train_chexpert
from train import train_chexpert

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    # data_dir = os.path.abspath("./data")
    # load_data(data_dir)
    ray_config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([128, 256, 512])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        train_chexpert,
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=ray_config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.ray_config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = DenseNet121(num_classes=3).to(device)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)