from ray.air import Result
from glob import glob
from ray import tune, air
from ray.tune.examples.mnist_pytorch import train_mnist
from ray.tune import ResultGrid
from train import *


class RayAnalysis:
    def __init__(self, result):
        self.result = result

        if type(self.result) == str:
            restored_tuner = tune.Tuner.restore(self.result)
            result_grid = restored_tuner.get_results()
        else:
            result_grid = self.result
        self.best_result: Result = result_grid.get_best_result(metric="val_roc_auc", mode="max")

    def get_best_checkpoint(self):
        best_dir = self.best_result.log_dir

        checkpoint_path = glob(os.path.join(best_dir, '*pth'))
        assert len(checkpoint_path) == 1
        best_checkpoint = checkpoint_path[0]

        return best_checkpoint
    
    def get_best_config(self): 
        best_config = self.best_result.config
        return best_config

    # etc.
